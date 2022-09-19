
import numbers
import os
from collections import defaultdict

import numpy as np
from scipy.special import gammaln, psi
from scipy.special import polygamma
from gensim import utils


class Lda_TopicModel():

    def __init__(self, corpus=None, num_topics=10, id2word=None,
                 batchsize=2000, epochs=1, update_every=1,
                 alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10,
                 iterations=50, gamma_threshold=0.001, minimum_probability=0.01,
                 ns_conf=None, minimum_phi_value=0.01,
                 per_word_topics=False, callbacks=None, dtype=np.float32):

        self.dtype = np.finfo(dtype).dtype

        self.id2word = id2word

        if len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0

        self.num_topics = int(num_topics)
        self.batchsize = batchsize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability
        self.num_updates = 0

        self.epochs = epochs
        self.update_every = update_every
        self.eval_every = eval_every
        self.minimum_phi_value = minimum_phi_value
        self.per_word_topics = per_word_topics
        self.callbacks = callbacks

        self.alpha, self.optimize_alpha = self.init_dir_prior(alpha, 'alpha')
        
        self.eta, self.optimize_eta = self.init_dir_prior(eta, 'eta')
        
        self.iterations= iterations
        self.gamma_threshold = gamma_threshold

        self.dispatcher = None
        self.numworkers = 1
        
        self.sstats = np.zeros((self.num_topics, self.num_terms), dtype=dtype)
        self.numdocs = 0


        # Initialize the variational distribution q(beta|lambda)
        self.sstats = np.random.gamma(100., 1. / 100., (self.num_topics, self.num_terms))
        self.expElogbeta = np.exp(dirichlet_expectation(self.sstats))

        # Check that we haven't accidentally fallen back to np.float64
        assert self.eta.dtype == self.dtype
        assert self.expElogbeta.dtype == self.dtype

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            use_numpy = self.dispatcher is not None
            self.update(corpus, batchs_as_numpy=use_numpy)

    def init_dir_prior(self, prior, name):
        
        if prior is None:
            prior = 'symmetric'

        if name == 'alpha':
            prior_shape = self.num_topics
        elif name == 'eta':
            prior_shape = self.num_terms


        is_auto = False

        if isinstance(prior, str):
            if prior == 'symmetric':
                init_prior = np.fromiter(
                    (1.0 / self.num_topics for i in range(prior_shape)),
                    dtype=self.dtype, count=prior_shape,
                )
            elif prior == 'asymmetric':
                if name == 'eta':
                    raise ValueError("The 'asymmetric' option cannot be used for eta")
                init_prior = np.fromiter(
                    (1.0 / (i + np.sqrt(prior_shape)) for i in range(prior_shape)),
                    dtype=self.dtype, count=prior_shape,
                )
                init_prior /= init_prior.sum()
                print("using asymmetric %s %s", name, list(init_prior))
            elif prior == 'auto':
                is_auto = True
                init_prior = np.fromiter((1.0 / self.num_topics for i in range(prior_shape)),
                    dtype=self.dtype, count=prior_shape)
                if name == 'alpha':
                    print("using autotuned %s, starting with %s", name, list(init_prior))
            else:
                raise ValueError("Unable to determine proper %s value given '%s'" % (name, prior))
        elif isinstance(prior, list):
            init_prior = np.asarray(prior, dtype=self.dtype)
        elif isinstance(prior, np.ndarray):
            init_prior = prior.astype(self.dtype, copy=False)
        elif isinstance(prior, (np.number, numbers.Real)):
            init_prior = np.fromiter((prior for i in range(prior_shape)), dtype=self.dtype)
        else:
            raise ValueError("%s must be either a np array of scalars, list of scalars, or scalar" % name)

        return init_prior, is_auto


    def sync_state(self, current_Elogbeta=None):
 
        if current_Elogbeta is None:
            current_Elogbeta = self.get_Elogbeta()
        self.expElogbeta = np.exp(current_Elogbeta)
        assert self.expElogbeta.dtype == self.dtype


    def inference(self, batch, collect_sstats=False):

        try:
            len(batch)
        except TypeError:
            batch = list(batch)

        gamma = np.random.gamma(100., 1. / 100., (len(batch), self.num_topics)).astype(self.dtype, copy=False)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        assert Elogtheta.dtype == self.dtype
        assert expElogtheta.dtype == self.dtype

        if collect_sstats:
            sstats = np.zeros_like(self.expElogbeta, dtype=self.dtype)
        else:
            sstats = None
        converged = 0

        integer_types = (int, np.integer,)
        epsilon = np.finfo(self.dtype).eps
        for d, doc in enumerate(batch):
            if len(doc) > 0 and not isinstance(doc[0][0], integer_types):
                ids = [int(idx) for idx, _ in doc]
            else:
                ids = [idx for idx, _ in doc]
            cts = np.fromiter((cnt for _, cnt in doc), dtype=self.dtype, count=len(doc))
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self.expElogbeta[:, ids]

            phinorm = np.dot(expElogthetad, expElogbetad) + epsilon

            for _ in range(self.iterations):
                lastgamma = gammad
                gammad = self.alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + epsilon
                meanchange = np.mean(np.abs(gammad - lastgamma))
                if meanchange < self.gamma_threshold:
                    converged += 1
                    break
            gamma[d, :] = gammad
            assert gammad.dtype == self.dtype
            if collect_sstats:
                sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)

        if collect_sstats:
            sstats *= self.expElogbeta
            assert sstats.dtype == self.dtype

        assert gamma.dtype == self.dtype
        return gamma, sstats

    def do_estep(self, batch):
 
        gamma, sstats = self.inference(batch, collect_sstats=True)
        self.sstats += sstats
        self.numdocs += gamma.shape[0]
        assert gamma.dtype == self.dtype
        return gamma

    def update_alpha(self, gammat, rho):

        N = float(len(gammat))
        logphat = sum(dirichlet_expectation(gamma) for gamma in gammat) / N
        assert logphat.dtype == self.dtype

        self.alpha = update_dir_prior(self.alpha, N, logphat, rho)

        assert self.alpha.dtype == self.dtype
        return self.alpha

    def update_eta(self, lambdat, rho):

        N = float(lambdat.shape[0])
        logphat = (sum(dirichlet_expectation(lambda_) for lambda_ in lambdat) / N).reshape((self.num_terms,))
        assert logphat.dtype == self.dtype

        self.eta = update_dir_prior(self.eta, N, logphat, rho)

        assert self.eta.dtype == self.dtype
        return self.eta

    def log_perplexity(self, batch, total_docs=None):

        if total_docs is None:
            total_docs = len(batch)
        corpus_words = sum(cnt for document in batch for _, cnt in document)
        subsample_ratio = 1.0 * total_docs / len(batch)
        perwordbound = self.bound(batch, subsample_ratio=subsample_ratio) / (subsample_ratio * corpus_words)
        return perwordbound

    def update(self, corpus, batchsize=None, decay=None, offset=None,
               epochs=None, update_every=None, eval_every=None, iterations=None,
               gamma_threshold=None, batchs_as_numpy=False):

        if decay is None:
            decay = self.decay
        if offset is None:
            offset = self.offset
        if epochs is None:
            epochs = self.epochs
        if update_every is None:
            update_every = self.update_every
        if eval_every is None:
            eval_every = self.eval_every
        if iterations is None:
            iterations = self.iterations
        if gamma_threshold is None:
            gamma_threshold = self.gamma_threshold

        try:
            lencorpus = len(corpus)
        except Exception:
            lencorpus = sum(1 for _ in corpus)
        if lencorpus == 0:
            return

        if batchsize is None:
            batchsize = min(lencorpus, self.batchsize)

        self.numdocs += lencorpus

        if update_every:
            updatetype = "online"
            if epochs == 1:
                updatetype += " (single-pass)"
            else:
                updatetype += " (multi-pass)"
            updateafter = min(lencorpus, update_every * self.numworkers * batchsize)
        else:
            updatetype = "batch"
            updateafter = lencorpus
        evalafter = min(lencorpus, (eval_every or 0) * self.numworkers * batchsize)

        updates_per_pass = max(1, lencorpus / updateafter)


        def rho():
            return pow(offset + pass_ + (self.num_updates / batchsize), -decay)



        for pass_ in range(epochs):

            reallen = 0
            batchs = utils.grouper(corpus, batchsize, as_numpy=batchs_as_numpy, dtype=self.dtype)
            for batch_no, batch in enumerate(batchs):
                reallen += len(batch)

                if eval_every and ((reallen == lencorpus) or ((batch_no + 1) % (eval_every * self.numworkers) == 0)):
                    self.log_perplexity(batch, total_docs=lencorpus)

                if self.dispatcher:
                    self.dispatcher.putjob(batch)
                else:
                    gammat = self.do_estep(batch)

                    if self.optimize_alpha:
                        self.update_alpha(gammat, rho())
                del batch

                if update_every and (batch_no + 1) % (update_every * self.numworkers) == 0:
                    if self.dispatcher:
                        other = self.dispatcher.getstate()
                    self.do_mstep(rho(), pass_ > 0)
                    

            # append current epoch's metric values
            if self.callbacks:
                current_metrics = callback.on_epoch_end(pass_)
                for metric, value in current_metrics.items():
                    self.metrics[metric].append(value)


    def do_mstep(self, rho, extra_pass=False):
        previous_Elogbeta = self.get_Elogbeta()
        self.blend(rho)

        current_Elogbeta = self.get_Elogbeta()
        self.sync_state(current_Elogbeta)

        if self.optimize_eta:
            self.update_eta(self.get_lambda(), rho)

        if not extra_pass:
            self.num_updates += self.numdocs

    def bound(self, corpus, gamma=None, subsample_ratio=1.0):
 
        score = 0.0
        _lambda = self.get_lambda()
        Elogbeta = dirichlet_expectation(_lambda)

        for d, doc in enumerate(corpus):
            if gamma is None:
                gammad, _ = self.inference([doc])
            else:
                gammad = gamma[d]
            Elogthetad = dirichlet_expectation(gammad)

            assert gammad.dtype == self.dtype
            assert Elogthetad.dtype == self.dtype

            # E[log p(doc | theta, beta)]
            score += sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)

            # E[log p(theta | alpha) - log q(theta | gamma)]; assumes alpha is a vector
            score += np.sum((self.alpha - gammad) * Elogthetad)
            score += np.sum(gammaln(gammad) - gammaln(self.alpha))
            score += gammaln(np.sum(self.alpha)) - gammaln(np.sum(gammad))

        score *= subsample_ratio

        # E[log p(beta | eta) - log q (beta | lambda)]; assumes eta is a scalar
        score += np.sum((self.eta - _lambda) * Elogbeta)
        score += np.sum(gammaln(_lambda) - gammaln(self.eta))

        if np.ndim(self.eta) == 0:
            sum_eta = self.eta * self.num_terms
        else:
            sum_eta = np.sum(self.eta)

        score += np.sum(gammaln(sum_eta) - gammaln(np.sum(_lambda, 1)))

        return score


    def blend(self, rhot, targetsize=None):

        if targetsize is None:
            targetsize = self.numdocs

        # stretch the current model's expected n*phi counts to target size
        if self.numdocs == 0 or targetsize == self.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / self.numdocs
        self.sstats *= (1.0 - rhot) * scale

        # stretch the incoming n*phi counts to target size
        if self.numdocs == 0 or targetsize == self.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / self.numdocs
        self.sstats += rhot * scale * self.sstats

        self.numdocs = targetsize

 
    def get_lambda(self):

        return self.eta + self.sstats

    def get_Elogbeta(self):
 
        return dirichlet_expectation(self.get_lambda())



def update_dir_prior(prior, N, logphat, rho):

    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)

    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)

    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))

    dprior = -(gradf - b) / q

    updated_prior = rho * dprior + prior
    if all(updated_prior > 0):
        prior = updated_prior
    else:
        print("updated prior is not positive")
    return prior

def logsumexp(x):

        x_max = np.max(x)
        x = np.log(np.sum(np.exp(x - x_max)))
        x += x_max

        return x

def dirichlet_expectation(alpha):

        if len(alpha.shape) == 1:
            result = psi(alpha) - psi(np.sum(alpha))
        else:
            result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
        return result.astype(alpha.dtype, copy=False)


