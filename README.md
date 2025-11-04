# Proteus Attention

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zen-Sherbert/Proteus-Attention/blob/main/TinyPlayground.ipynb)

I want to start by thanking you for visiting and taking an interest in my project.

To get right to the point: this is an experimental attention architecture. The entire goal of this system is to enable **extreme long context on humble devices.**

Are you tired of seeing new models with million-token context windows that require a mountain of H100s to run? Did you, like me, not get a small loan of a million dollars to build a GPU cluster?

> *(I know this sounds like a sales pitch. I've been up all night fueled by coffee and decided it was the best time to make myself useful.)*

Look no further. I got you covered.

> ### Why use a hundred GPUs, when one is good enough?

---

## The Philosophy: The Life of a Humble Token

To understand how Proteus works, let's forget about the complex math for a second and look at it from the eyes of a single word.

**Our story begins.** We are a lonely token, the word **"Potato,"** trapped on page 55,555 of some godforsaken document. We're just straight chilling, surrounded by millions of other, less interesting (but still cool) words.

For most AI models, we are invisible, lost in the noise. But for Proteus, our story is just beginning.

**1. The User:** A user, at the very end of the document, asks: *"How do you make potato soup?"*

**2. DNA (cue your favorite CSI show):** Suddenly, a signal flashes through the entire document. The DNA system, a set of "salience gates" that have learned the "flavor" of important concepts, is activated. A gate that has evolved to recognize "food" and "recipes" sees us. It gives our "Potato" token a massive "importance score." We are no longer just a word; we are a **clue.**
<br>
>*(This was literally a random add-on that just kind of made itself more and more useful.)*

**3. Chunking (Yes, that chunking):** The document is being read in "chunks" of a few thousand tokens at a time. Our "Potato" token is now in a local arena, competing with the other words in its immediate neighborhood. Because of its high score, it easily wins a spot as one of the "champions" of its chunk. It gets promoted.
<br>
>*(ML is not my original arena; game design is. So this comes from that.)*

**4. The Buffer:** This is the big one. Our "Potato" token is now sent to the main stage, a fixed-size buffer that acts as a "Hall of Heroes." Here, it must compete with the champions from *every other chunk* in the entire document. A weak word like "therefore" from page 100 is kicked out to make room. Our "Potato," being a key part of the answer, earns its permanent spot.
<br>
>*(This is what gives us decent info across a massive document. You can set it to whatever size your GPU can balance between the active chunk and buffer.)*

**5. Teleportation:** The final reasoning pass begins. The model is not looking at the full, noisy document. It is looking at the curated "Hall of Heroes." In this small, high-signal space, it sees the user's query ("potato soup") and our "Potato" token in the same view. Thanks to **RoPE**, it knows our token came from page 55,555.

The system has just created an **Einstein-Rosen Bridge** through the document, connecting the query at the end to our lonely "Potato" token from the distant past. It has teleported the answer to the user.

Wild, I know.
>*(This seems far-fetched, and I literally had to slam my head into my keyboard to understand this. Percussive osmosis does not work.)*

---

## Don't Trust Me. Prove It to Yourself.

Talk is cheap. The only thing that matters is proof.

This entire project is packaged into a single Google Colab notebook. You don't need a powerful machine; you just need a web browser. In it, you will find:

*   **The Live Benchmark:** Run a head-to-head comparison against standard attention on a free T4 GPU and watch it break the memory wall.
*   **The Validation Suite:** Run the full set of "Needle in a Haystack" and "Jigsaw Puzzle" tests that prove the mechanics are sound.
*   **The Scaling Demo:** Watch the Flux Chunking system chew through millions of tokens with a tiny, fixed memory footprint.

**[► Click here to open the Proteus Playground in Google Colab](https://colab.research.google.com/github/Zen-Sherbert/Proteus-Attention/blob/main/TinyPlayground.ipynb)**

---

## Quickstart

If you want to run it on your own machine:

```bash
# 1. Clone the repo
git clone https://github.com/Zen-Sherbert/Proteus-Attention.git

# 2. CD into the directory
cd Proteus-Attention

# 3. Install the package
pip install .

# 4. Run the tests and benchmarks from the /scripts folder!
python scripts/chunked_flux_tests.py
```

---

## About This Project

If you're wondering what brought about this mess, you may look here.

I built this on my 7800XT with 16GB of VRAM because I was told that it was impossible to run a model of a certain size with a context window of a certain size.

I did not like that.

My time is a precious commodity, and so is money. I do not have the resources of a major lab. I was never gifted a small loan of a million dollars.

So to put it all into perspective, this was made in the middle of the night during my shift as an overnight asset protection officer. It was made while bouncing my two-year-old daughter in my lap, while she donated her own additions to my system in the form of slapping my keyboard.

If anyone else ends up seeing this, do not be intimidated by a challenge. The odds will always be stacked one way or the other.

I am a 32-year-old veteran, a father of three, working like a vampire in the middle of the night, with no formal education in ML.

So if I can make this, you can do anything infinitely better.
