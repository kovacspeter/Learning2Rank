import numpy as np

class ESRank:

    """
        ES-Rank: Evolution Strategy Learning to Rank Approach
            - http://www.cs.nott.ac.uk/~psxoi/dls_sac2017.pdf

        Usage example:
            esrank = ESRank(eval_fun).fit(X, y, n_generations)

            ranked = esrank()
    """
    def __init__(self, eval_fun):
        self.eval_fun = eval_fun #function to maximize with signature eval_fun(predicted, true)

    def predict(self, features):
        return self._predict(features, self.params)

    def _predict(self, features, params):
        return features * params


    def fit(self, X, y, n_generations=1500):
        self.n_genes = X.shape[1]
        self.params = np.zeros(self.n_genes)
        new_params = np.zeros(self.n_genes)

        good = False

        for g in range(n_generations):
            # MUTATE
            if good:
                new_params = self._mutation(self.params, True)
            else:
                new_params = self._mutation(self.params, False)

            # EVALUATE
            # TODO evaluation takes too much time (evaluate only on small subset and 'estimate' fitness)
            if self.eval_fun(y, self._predict(X, self.params)) < self.eval_fun(y, self._predict(X, new_params)):
                self.params = new_params
                good = True
            else:
                new_params = self.params
                good = False

        return self

    def _mutation(self, vector, repeat=False):
        if not repeat:
            self.mask = np.random.randint(2, size=self.n_genes)
            self.mutations = np.random.normal(0, 1, self.n_genes) * np.exp(np.random.standard_cauchy(self.n_genes))

        return vector + self.mutations * self.mask

