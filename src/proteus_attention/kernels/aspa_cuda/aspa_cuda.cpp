#include <ATen/ATen.h>
#include <ATen/ops/arange_ops.h>
#include <ATen/ops/bincount_ops.h>
#include <ATen/ops/cumsum_ops.h>
#include <ATen/ops/dropout_ops.h>
#include <ATen/ops/index_select_ops.h>
#include <ATen/ops/masked_fill_ops.h>
#include <ATen/ops/scaled_dot_product_attention_ops.h>
#include <ATen/ops/softmax_ops.h>
#include <ATen/ops/zeros_like_ops.h>
#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cmath>
#include <limits>
#include <string>
#include <tuple>

namespace py = pybind11;
using torch::Tensor;
using torch::indexing::Slice;

namespace {

void record_backend(const std::string& name, const py::dict& details) {
    try {
        py::module sparse = py::module::import("proteus_attention.kernels.sparse_attn");
        py::object fn = sparse.attr("_record_backend");
        fn(name, **details);
    } catch (const py::error_already_set&) {
        PyErr_Clear();
    } catch (...) {
        // Swallow errors so the CUDA path degrades gracefully.
    }
}

struct PackedRows {
    Tensor head_idx;
    Tensor token_idx;
    Tensor row_offsets;
    int64_t max_rows{0};
    bool valid{false};
};

PackedRows pack_active_rows(const Tensor& mask) {
    PackedRows packed;
    auto device = mask.device();
    const int64_t total_heads = mask.size(0);
    auto active_positions = mask.nonzero();
    if (active_positions.numel() == 0) {
        return packed;
    }
    packed.valid = true;
    packed.head_idx =
        active_positions.index({Slice(), 0}).to(torch::kLong).contiguous();
    packed.token_idx =
        active_positions.index({Slice(), 1}).to(torch::kLong).contiguous();
    auto counts = torch::bincount(
        packed.head_idx,
        /*weights=*/Tensor(),
        /*minlength=*/total_heads);
    counts = counts.to(torch::kInt32);
    auto offsets_options =
        torch::TensorOptions().device(device).dtype(torch::kInt32);
    packed.row_offsets =
        torch::zeros({total_heads + 1}, offsets_options);
    packed.row_offsets.index_put_(
        {Slice(1, torch::indexing::None)},
        torch::cumsum(counts, 0, torch::kInt32));
    packed.max_rows =
        counts.numel() > 0 ? counts.max().item<int64_t>() : 0;
    return packed;
}

PackedRows dense_pack_rows(int64_t heads, int64_t tokens, const torch::Device& device) {
    PackedRows packed;
    auto long_opts =
        torch::TensorOptions().device(device).dtype(torch::kLong);
    auto int_opts =
        torch::TensorOptions().device(device).dtype(torch::kInt32);
    packed.head_idx =
        torch::arange(heads, long_opts).repeat_interleave(tokens);
    packed.token_idx =
        torch::arange(tokens, long_opts).repeat({heads});
    packed.row_offsets = torch::arange(
        0,
        (heads + 1) * tokens,
        tokens,
        int_opts);
    packed.max_rows = tokens;
    packed.valid = true;
    return packed;
}

Tensor ensure_bool_mask(const c10::optional<Tensor>& mask_opt,
                        int64_t heads,
                        int64_t tokens,
                        const torch::Device& device) {
    auto bool_opts =
        torch::TensorOptions().device(device).dtype(torch::kBool);
    if (!mask_opt.has_value()) {
        return torch::ones({heads, tokens}, bool_opts);
    }
    Tensor mask = mask_opt.value();
    if (mask.dim() == 3 && mask.size(2) == 1) {
        mask = mask.squeeze(-1);
    }
    if (mask.scalar_type() != torch::kBool) {
        mask = mask > 0;
    }
    return mask.to(bool_opts).contiguous();
}

}  // namespace

Tensor aspa_sparse_attention(
    Tensor q,
    Tensor k,
    Tensor v,
    c10::optional<Tensor> active_mask,
    c10::optional<Tensor> causal_mask,
    double dropout_p,
    bool training,
    py::object prepacked,
    c10::optional<Tensor> flux_candidates,
    c10::optional<Tensor> flux_lengths) {
    TORCH_CHECK(q.dim() == 3, "q must have rank 3");
    TORCH_CHECK(k.dim() == 3, "k must have rank 3");
    TORCH_CHECK(v.dim() == 3, "v must have rank 3");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "q, k, v must share shapes");

    const auto device = q.device();
    const int64_t batch_heads = q.size(0);
    const int64_t tokens = q.size(1);
    const int64_t head_dim = q.size(2);
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    auto mask_bool = ensure_bool_mask(active_mask, batch_heads, tokens, device);

    PackedRows packed;
    if (!prepacked.is_none()) {
        auto tuple_obj = prepacked.cast<py::tuple>();
        TORCH_CHECK(tuple_obj.size() == 4, "prepacked must be a 4-tuple");
        packed.head_idx = tuple_obj[0].cast<Tensor>().to(torch::kLong).contiguous();
        packed.token_idx = tuple_obj[1].cast<Tensor>().to(torch::kLong).contiguous();
        packed.row_offsets =
            tuple_obj[2].cast<Tensor>().to(torch::kInt32).contiguous();
        packed.max_rows = tuple_obj[3].cast<int64_t>();
        packed.valid = true;
    } else if (mask_bool.all().item<bool>()) {
        packed = dense_pack_rows(batch_heads, tokens, device);
    } else {
        packed = pack_active_rows(mask_bool);
        if (!packed.valid) {
            auto zeros = torch::zeros_like(q);
            py::dict info;
            info["device"] = py::str(q.device().str());
            info["heads"] = py::int_(batch_heads);
            info["quantized"] = py::bool_(false);
            record_backend("empty", info);
            return zeros;
        }
    }

    TORCH_CHECK(packed.valid, "Packed rows must be available");

    Tensor flux_cand_tensor;
    Tensor flux_len_tensor;
    bool use_flux = false;
    if (flux_candidates.has_value() && flux_lengths.has_value()) {
        flux_cand_tensor =
            flux_candidates.value().to(torch::kLong).to(device).contiguous();
        flux_len_tensor =
            flux_lengths.value().to(torch::kLong).to(device).contiguous();
        if (flux_cand_tensor.numel() > 0) {
            TORCH_CHECK(
                flux_cand_tensor.size(0) == packed.token_idx.size(0),
                "Flux candidate rows must match active token count");
            TORCH_CHECK(
                flux_len_tensor.size(0) == packed.token_idx.size(0),
                "Flux candidate lengths must match active token count");
            use_flux = true;
        }
    }

    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();
    auto v_contig = v.contiguous();

    auto q_view = q_contig.view({batch_heads, tokens, head_dim});
    auto k_view = k_contig.view({batch_heads, tokens, head_dim});
    auto v_view = v_contig.view({batch_heads, tokens, head_dim});

    auto q_float = q_view.scalar_type() == torch::kFloat32
                       ? q_view
                       : q_view.to(torch::kFloat32);
    auto k_float = k_view.scalar_type() == torch::kFloat32
                       ? k_view
                       : k_view.to(torch::kFloat32);
    auto v_float = v_view.scalar_type() == torch::kFloat32
                       ? v_view
                       : v_view.to(torch::kFloat32);

    auto result_opts =
        torch::TensorOptions().device(device).dtype(torch::kFloat32);
    auto result =
        torch::zeros({batch_heads, tokens, head_dim}, result_opts);

    Tensor causal = causal_mask.has_value()
                        ? causal_mask.value().to(device).to(torch::kFloat32)
                        : Tensor();

    auto row_offsets_cpu = packed.row_offsets.to(torch::kCPU);
    const int64_t total_active = packed.token_idx.size(0);

    for (int64_t head = 0; head < batch_heads; ++head) {
        const int64_t start =
            row_offsets_cpu.index({head}).item<int64_t>();
        const int64_t end =
            row_offsets_cpu.index({head + 1}).item<int64_t>();
        if (start == end) {
            continue;
        }
        auto rows_tokens =
            packed.token_idx.index({Slice(start, end)});
        auto rows_tokens_long = rows_tokens.to(torch::kLong);
        const int64_t row_count = rows_tokens_long.size(0);
        auto q_head = q_float.index({head});
        auto k_head = k_float.index({head});
        auto v_head = v_float.index({head});
        auto q_sel = q_head.index_select(0, rows_tokens_long);

        if (use_flux) {
            auto cand_rows =
                flux_cand_tensor.index({Slice(start, end)});
            auto len_rows =
                flux_len_tensor.index({Slice(start, end)});
            const int64_t max_len = cand_rows.size(1);
            auto clamped =
                cand_rows.clamp(0, std::max<int64_t>(tokens - 1, 0));
            auto flat_idx = clamped.reshape({-1});
            auto k_sel = k_head.index_select(0, flat_idx)
                             .view({row_count, max_len, head_dim});
            auto v_sel = v_head.index_select(0, flat_idx)
                             .view({row_count, max_len, head_dim});
            auto mask_invalid = torch::arange(
                                    max_len,
                                    torch::TensorOptions()
                                        .device(device)
                                        .dtype(torch::kLong))
                                    .unsqueeze(0) >=
                                len_rows.unsqueeze(1);
            k_sel.masked_fill_(mask_invalid.unsqueeze(-1), 0.0);
            v_sel.masked_fill_(mask_invalid.unsqueeze(-1), 0.0);
            auto scores = torch::bmm(
                              q_sel.unsqueeze(1),
                              k_sel.transpose(1, 2))
                              .squeeze(1);
            scores.mul_(scale);
            if (causal_mask.has_value()) {
                auto causal_rows =
                    causal.index_select(0, rows_tokens_long);
                auto causal_clamped = clamped.clamp(
                    0,
                    std::max<int64_t>(causal.size(1) - 1, 0));
                auto causal_slice =
                    torch::gather(
                        causal_rows,
                        1,
                        causal_clamped.to(torch::kLong));
                scores = scores + causal_slice;
            }
            scores.masked_fill_(mask_invalid, -std::numeric_limits<float>::infinity());
            auto probs = torch::softmax(scores, -1);
            if (training && dropout_p > 0.0) {
                probs = torch::dropout(probs, dropout_p, true);
            }
            auto out_sel = torch::bmm(
                               probs.unsqueeze(1),
                               v_sel)
                               .squeeze(1);
            result.index_put_({head, rows_tokens_long}, out_sel);
        } else {
            const int64_t max_token =
                rows_tokens_long.max().item<int64_t>();
            auto upper = std::min<int64_t>(max_token + 1, tokens);
            auto k_slice = k_head.index({Slice(0, upper)});
            auto v_slice = v_head.index({Slice(0, upper)});
            c10::optional<Tensor> attn_mask = c10::nullopt;
            if (causal_mask.has_value()) {
                auto mask_rows =
                    causal.index_select(0, rows_tokens_long);
                mask_rows = mask_rows.index({Slice(), Slice(0, upper)});
                attn_mask = mask_rows.unsqueeze(0);
            } else {
                auto key_positions = torch::arange(
                    upper,
                    torch::TensorOptions()
                        .device(device)
                        .dtype(rows_tokens_long.scalar_type()));
                auto future_mask =
                    key_positions.unsqueeze(0) >
                    rows_tokens_long.unsqueeze(1);
                if (future_mask.any().item<bool>()) {
                    auto attn = torch::zeros(
                        {1, row_count, upper},
                        torch::TensorOptions()
                            .device(device)
                            .dtype(torch::kFloat32));
                    attn = attn.masked_fill(
                        future_mask.unsqueeze(0),
                        -std::numeric_limits<float>::infinity());
                    attn_mask = attn;
                }
            }
            auto out_sel = at::scaled_dot_product_attention(
                q_sel.unsqueeze(0),
                k_slice.unsqueeze(0),
                v_slice.unsqueeze(0),
                attn_mask,
                c10::nullopt,
                dropout_p,
                /*is_causal=*/false,
                c10::nullopt);
            out_sel = out_sel.squeeze(0);
            result.index_put_({head, rows_tokens_long}, out_sel);
        }
    }

    auto output = result.to(q.scalar_type()).reshape_as(q);

    py::dict info;
    info["device"] = py::str(q.device().str());
    info["heads"] = py::int_(batch_heads);
    info["tokens"] = py::int_(tokens);
    info["max_rows"] = py::int_(packed.max_rows);
    record_backend("cuda", info);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "aspa_sparse_attention",
        &aspa_sparse_attention,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("active_mask") = py::none(),
        py::arg("causal_mask") = py::none(),
        py::arg("dropout_p") = 0.0,
        py::arg("training") = false,
        py::arg("prepacked") = py::none(),
        py::arg("flux_candidates") = py::none(),
        py::arg("flux_lengths") = py::none(),
        "CUDA-backed sparse attention fallback for ASPA.");
}
