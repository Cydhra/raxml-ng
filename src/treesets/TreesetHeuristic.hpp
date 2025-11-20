#ifndef RAXML_TREESETHEURISTIC_HPP_
#define RAXML_TREESETHEURISTIC_HPP_

/**
 * An aggressive stateful local search heuristic for treeset-search. This differs substantially from other topological
 * heuristics because those are stateless (i.e., they do not depend on previous results of the same raxml search).
 * This heuristic auto-tunes its search parameters in between batches of tree searches to reduce the amount of effort
 * spent on individual searches as much as possible.
 * This does not lead to a deterioration of the final tree accuracy under the assumption that the treeset command is
 * used for degenerate datasets with extremely low signal.
 * If the likelihood surface has pronounced peaks, this aggressive heuristic will likely not yield optimal results (but
 * this will be detectable by the AU test).
 * 
 */
class TreesetHeuristic {
public:
	/**
	 * @param num_spr the number of SPR rounds used by the standard heuristic as a starting point of auto-tuning.
	 */
	explicit TreesetHeuristic(const unsigned int num_spr) : light_spr(false), skip_model(false), num_spr(num_spr) {}

private:
	/**
	 * If true, replace fast SPR rounds with light SPR rounds that do even less BLOs.
	 */
	bool light_spr;

	/**
	 * If true, skip the first model optimization by reusing model parameters from the previous search.
	 */
	bool skip_model;

	/**
	 * How many SPR rounds to perform for each tree search
	 */
	unsigned int num_spr;
};

/**
 * A batch of local searches tuned with a specific set of parameters chosen by a {@link TreesetHeuristic} instance.
 */
class TunedBatch final {
public:
	TunedBatch(const bool light_spr, const bool skip_model, const unsigned int num_spr)
		: light_spr(light_spr),
		  skip_model(skip_model),
		  num_spr(num_spr) {
	}

	/**
	 * If true, replace fast SPR rounds with light SPR rounds that do even less BLOs.
	 */
	const bool light_spr;

	/**
	 * If true, skip the first model optimization by reusing model parameters from the previous search.
	 */
	const bool skip_model;

	/**
	 * How many SPR rounds to perform for each tree search
	 */
	const unsigned int num_spr;

private:

};

#endif //RAXML_TREESETHEURISTIC_HPP_