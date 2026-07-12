Read `.agents/AGENTS.md` and `agent_notes.md` for some background about the project and its current state. Your goal will be to modify or potentially fully overhaul the algorithm for determining optimal glide path (up to horizon 50-- don't worry about the retirement arm yet). Both the greedy and bisection approaches don't seem to arrive anywhere close to the global optimum, and I'm out of ideas for computationally efficient approaches that can produce provably good solutions.

Here are some hard constraints:

- We'll stick with real returns on the existing three asset classes, using the current dataset only
- Continue to generally use block-bootstrapping for simulating returns
- Maintain the general assumption that a glide path should exhibit a constant level of risk at each horizon, and that risk should generally optimize bad-case outcomes. I've done that by trying to optimize the mean among the worst 4% of outcomes at all horizons, but feel free to play around with metric used. I also opted to summarize a full path's objective as the mean over the metric value across horizons, but that can be adjusted if needed as well

Otherwise, experiment freely. In `evaluate_greedy_algorithm` you'll see that I've written some basic heuristics for sanity-checking the path quality: extending/contracting it, switching assets, or linearizing the path should obviously not improve the optimization metric. These are not particularly robust proof that we're getting close to the global optimum, so please develop your own tests that will convince me you've found a great glide path. If the surface you're optimizing turns out to be noisy and you need to smooth/regularize/adjust it in any way, please also produce plots or other proof that smoothing parameters, etc, are necessary and well-chosen.

Operate agentically as needed until you've solved the problem, and please place all your code and plots in a separate directory to keep things isolated and independent for now.
