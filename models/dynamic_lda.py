
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize

from gensim import utils
from models import lda_topicmodel

class Dynamic_LdaModel():

    def __init__(
            self, corpus=None, articles_per_time=None, id2word=None, alphas=0.01, num_topics=10,
            sstats=None, approx_true_fwd_var=0.5, beta_variance=0.005, epochs=10, lda_inference_max_epoch=25,  em_max_epoch=20, batchsize=100,
        ):
        
        self.id2word = id2word
        self.vocab_len = len(self.id2word)
        self.corpus_len = len(corpus)
        self.articles_per_time = articles_per_time
        self.num_articles_per_times = len(articles_per_time)

        self.num_topics = num_topics
        self.num_articles_per_times = len(articles_per_time)
        self.alphas = np.full(num_topics, alphas)
        self.max_doc_len = max(len(line) for line in corpus)


        self.topic_chains = []
        for topic in range(num_topics):
            state_space_ = state_space(
                num_articles_per_times=self.num_articles_per_times, vocab_len=self.vocab_len, num_topics=self.num_topics,
                beta_variance=beta_variance, approx_true_fwd_var=approx_true_fwd_var
            )
            self.topic_chains.append(state_space_)
        
        self.max_doc_len = max(len(line) for line in corpus)

        lda_model = lda_topicmodel.Lda_TopicModel(corpus, id2word=self.id2word, num_topics=self.num_topics,
                                    epochs=epochs, alpha=self.alphas,
                                    dtype=np.float64)

        self.sstats = np.transpose(lda_model.sstats)

        # initialize model from sstats
        for k, chain in enumerate(self.topic_chains):
            k_sstats = self.sstats[:, k]
            state_space.state_space_counts_init(chain, approx_true_fwd_var, beta_variance, k_sstats)

        # fit DTM
        self.fit_dyn_lda(corpus, lda_inference_max_epoch, em_max_epoch, batchsize)


    def update_phi(self, doc_number, time):
        
        num_topics = self.lda.num_topics
        # digamma values
        dig = np.zeros(num_topics)

        for k in range(num_topics):
            dig[k] = digamma(self.gamma[k])

        n = 0   # phi, log_phi counter
        for word_id, count in self.doc:
            for k in range(num_topics):
                self.log_phi[n][k] = dig[k] + self.lda.topics[word_id][k]

            log_phi_row = self.log_phi[n]
            phi_row = self.phi[n]

            # log normalize
            v = log_phi_row[0]
            for i in range(1, len(log_phi_row)):
                v = np.logaddexp(v, log_phi_row[i])

            # subtract every element by v
            log_phi_row = log_phi_row - v
            phi_row = np.exp(log_phi_row)
            self.log_phi[n] = log_phi_row
            self.phi[n] = phi_row
            n += 1  # increase epochation

        return self.phi, self.log_phi

    def update_gamma(self):
        
        self.gamma = np.copy(self.lda.alpha)
        n = 0  #  phi, log_phi counter
        for word_id, count in self.doc:
            phi_row = self.phi[n]
            for k in range(self.lda.num_topics):
                self.gamma[k] += phi_row[k] * count
            n += 1

        return self.gamma

    def init_lda_post(self):
        total = sum(count for word_id, count in self.doc)
        self.gamma.fill(self.lda.alpha[0] + float(total) / self.lda.num_topics)
        self.phi[:len(self.doc), :] = 1.0 / self.lda.num_topics


    def compute_lda_lhood(self):
        
        num_topics = self.lda.num_topics
        gamma_sum = np.sum(self.gamma)


        lhood = gammaln(np.sum(self.lda.alpha)) - gammaln(gamma_sum)
        self.lhood[num_topics] = lhood

        # influence_term = 0
        digsum = digamma(gamma_sum)

        for k in range(num_topics):

            e_log_theta_k = digamma(self.gamma[k]) - digsum
            lhood_term = \
                (self.lda.alpha[k] - self.gamma[k]) * e_log_theta_k + \
                gammaln(self.gamma[k]) - gammaln(self.lda.alpha[k])

            n = 0
            for word_id, count in self.doc:
                if self.phi[n][k] > 0:
                    lhood_term += \
                        count * self.phi[n][k] * (e_log_theta_k + self.lda.topics[word_id][k] - self.log_phi[n][k])
                n += 1
            self.lhood[k] = lhood_term
            lhood += lhood_term


        return lhood

    def fit_lda_post(self, doc_number, time, ldaseq, LDA_INFERENCE_CONVERGED=1e-8,lda_inference_max_epoch=25):
    

        self.init_lda_post()
        # sum of counts in a doc
        total = sum(count for word_id, count in self.doc)


        lhood = self.compute_lda_lhood()
        lhood_old = 0
        converged = 0
        epoch_ = 0

        # first epochation starts here
        epoch_ += 1
        lhood_old = lhood
        self.gamma = self.update_gamma()


        if state_space is None:
            self.phi, self.log_phi = self.update_phi(doc_number, time)

        lhood = self.compute_lda_lhood()
        converged = np.fabs((lhood_old - lhood) / (lhood_old * total))

        while converged > LDA_INFERENCE_CONVERGED and epoch_ <= lda_inference_max_epoch:

            epoch_ += 1
            lhood_old = lhood
            self.gamma = self.update_gamma()

            if state_space is None:
                self.phi, self.log_phi = self.update_phi(doc_number, time)

            lhood = self.compute_lda_lhood()
            converged = np.fabs((lhood_old - lhood) / (lhood_old * total))

        return lhood

    def update_dyn_lda_ss(self, time, doc, topic_suffstats):

        num_topics = self.lda.num_topics

        for k in range(num_topics):
            topic_ss = topic_suffstats[k]
            n = 0
            for word_id, count in self.doc:
                topic_ss[word_id][time] += count * self.phi[n][k]
                n += 1
            topic_suffstats[k] = topic_ss

        return topic_suffstats


    def fit_dyn_lda(self, corpus, lda_inference_max_epoch, em_max_epoch, batchsize):

        CONVERGENCE_THRESHOLD = 1e-4

        num_topics = self.num_topics
        vocab_len = self.vocab_len
        data_len = self.num_articles_per_times
        corpus_len = self.corpus_len

        bound = 0
        convergence = CONVERGENCE_THRESHOLD + 1
        epoch_ = 0

        while ((convergence > CONVERGENCE_THRESHOLD) and epoch_ <= em_max_epoch):
            # TODO: bound is initialized to 0
            old_bound = bound

            # initiate sufficient statistics
            topic_suffstats = []
            for topic in range(num_topics):
                topic_suffstats.append(np.zeros((vocab_len, data_len)))

            # set up variables
            gammas = np.zeros((corpus_len, num_topics))
            lhoods = np.zeros((corpus_len, num_topics + 1))
            # compute the likelihood of a sequential corpus under an LDA
            # seq model and find the evidence lower bound. This is the E - Step
            bound, gammas = \
                self.dyn_lda_infer(corpus, topic_suffstats, gammas, lhoods, epoch_, lda_inference_max_epoch, batchsize)
            self.gammas = gammas

            # fit the variational distribution. This is the M - Step
            topic_bound = self.fit_dyn_lda_topics(topic_suffstats)
            bound += topic_bound

            # check for convergence
            if(old_bound):
                convergence = np.fabs((bound - old_bound) / old_bound)

            epoch_ += 1

        return bound

    def dyn_lda_infer(self, corpus, topic_suffstats, gammas, lhoods,
                      epoch_, lda_inference_max_epoch, batchsize):

        num_topics = self.num_topics
        vocab_len = self.vocab_len
        bound = 0.0

        lda = lda_topicmodel.Lda_TopicModel(num_topics=num_topics, alpha=self.alphas, id2word=self.id2word, dtype=np.float64)
        lda.topics = np.zeros((vocab_len, num_topics))

        #ldapost = LdaPost(max_doc_len=self.max_doc_len, num_topics=num_topics, lda=lda)
        self.doc = None
        self.lda = lda
        self.lhood = lhoods
        self.gamma = gammas

        if self.gamma is None:
            self.gamma = np.zeros(num_topics)
        if self.lhood is None:
            self.lhood = np.zeros(num_topics + 1)

        if self.max_doc_len is not None and num_topics is not None:
            self.phi = np.zeros((self.max_doc_len, num_topics))
            self.log_phi = np.zeros((self.max_doc_len, num_topics))

        bound, gammas = self.infer_DTM(
            corpus, topic_suffstats, gammas, lhoods, lda,
            epoch_, bound, lda_inference_max_epoch, batchsize
        )


        return bound, gammas

    def infer_DTM(self, corpus, topic_suffstats, gammas, lhoods, lda,
                    epoch_, bound, lda_inference_max_epoch, batchsize):
 
        doc_index = 0  # doc_index in corpus
        time = 0  # current time-slice
        doc_num = 0  # doc-index in current time-slice
        lda = self.make_dyn_lda_slice(lda, time)  # create dyn_lda slice

        articles_per_time = np.cumsum(np.array(self.articles_per_time))

        for batch_no, batch in enumerate(utils.grouper(corpus, batchsize)):
            for doc in batch:
                if doc_index > articles_per_time[time]:
                    time += 1
                    lda = self.make_dyn_lda_slice(lda, time)  # create dyn_lda slice
                    doc_num = 0

                gam = gammas[doc_index]
                lhood = lhoods[doc_index]

                self.gamma = gam
                self.lhood = lhood
                self.doc = doc

                if epoch_ == 0:
                    doc_lhood = self.fit_lda_post(
                        doc_num, time, None, lda_inference_max_epoch=lda_inference_max_epoch
                    )
                else:
                    doc_lhood = self.fit_lda_post(
                        doc_num, time, self, lda_inference_max_epoch=lda_inference_max_epoch
                    )

                if topic_suffstats is not None:
                    topic_suffstats = self.update_dyn_lda_ss(time, doc, topic_suffstats)

                gammas[doc_index] = self.gamma
                bound += doc_lhood
                doc_index += 1
                doc_num += 1

        return bound, gammas

    def make_dyn_lda_slice(self, lda, time):

        for k in range(self.num_topics):
            lda.topics[:, k] = self.topic_chains[k].e_log_prob[:, time]

        lda.alpha = np.copy(self.alphas)
        return lda

    def fit_dyn_lda_topics(self, topic_suffstats):

        lhood = 0

        for k, chain in enumerate(self.topic_chains):
            lhood_term = state_space.fit_state_space(chain, topic_suffstats[k])
            lhood += lhood_term

        return lhood

 
    def get_topic(self, topic, time=0, top_terms=20):

        topic = self.topic_chains[topic].e_log_prob
        topic = np.transpose(topic)
        topic = np.exp(topic[time])
        topic = topic / topic.sum()
        bestn  = np.argpartition(topic, -int(top_terms))[-int(top_terms):]
        beststr = [(self.id2word[id_], topic[id_]) for id_ in bestn]
        return beststr


class state_space():

    def __init__(self, vocab_len=None, num_articles_per_times=None, num_topics=None, approx_true_fwd_var=0.5, beta_variance=0.005):

        self.vocab_len = vocab_len
        self.num_articles_per_times = num_articles_per_times
        self.approx_true_fwd_var = approx_true_fwd_var
        self.beta_variance = beta_variance
        self.num_topics = num_topics
        self.obs = np.zeros((vocab_len, num_articles_per_times))
        self.e_log_prob = np.zeros((vocab_len, num_articles_per_times))
        self.mean = np.zeros((vocab_len, num_articles_per_times + 1))
        self.fwd_mean = np.zeros((vocab_len, num_articles_per_times + 1))
        self.fwd_variance = np.zeros((vocab_len, num_articles_per_times + 1))
        self.variance = np.zeros((vocab_len, num_articles_per_times + 1))
        self.zeta = np.zeros(num_articles_per_times)


    def update_zeta(self):
        
        for j, val in enumerate(self.zeta):
            self.zeta[j] = np.sum(np.exp(self.mean[:, j + 1] + self.variance[:, j + 1] / 2))
        return self.zeta

    def compute_post_variance(self, word, beta_variance):

        INIT_VARIANCE_CONST = 1000

        T = self.num_articles_per_times
        variance = self.variance[word]
        fwd_variance = self.fwd_variance[word]
        # forward pass. Set initial variance very high
        fwd_variance[0] = beta_variance * INIT_VARIANCE_CONST
        for t in range(1, T + 1):
            if self.approx_true_fwd_var:
                c = self.approx_true_fwd_var / (fwd_variance[t - 1] + beta_variance + self.approx_true_fwd_var)
            else:
                c = 0
            fwd_variance[t] = c * (fwd_variance[t - 1] + beta_variance)

        # backward pass
        variance[T] = fwd_variance[T]
        for t in range(T - 1, -1, -1):
            if fwd_variance[t] > 0.0:
                c = np.power((fwd_variance[t] / (fwd_variance[t] + beta_variance)), 2)
            else:
                c = 0
            variance[t] = (c * (variance[t + 1] - beta_variance)) + ((1 - c) * fwd_variance[t])

        return variance, fwd_variance

    def compute_post_mean(self, word, beta_variance):

  
        T = self.num_articles_per_times
        obs = self.obs[word]
        fwd_variance = self.fwd_variance[word]
        mean = self.mean[word]
        fwd_mean = self.fwd_mean[word]

        # forward
        fwd_mean[0] = 0
        for t in range(1, T + 1):
            c = self.approx_true_fwd_var / (fwd_variance[t - 1] + beta_variance + self.approx_true_fwd_var)
            fwd_mean[t] = c * fwd_mean[t - 1] + (1 - c) * obs[t - 1]

        # backward pass
        mean[T] = fwd_mean[T]
        for t in range(T - 1, -1, -1):
            if beta_variance == 0.0:
                c = 0.0
            else:
                c = beta_variance / (fwd_variance[t] + beta_variance)
            mean[t] = c * fwd_mean[t] + (1 - c) * mean[t + 1]
        return mean, fwd_mean

    def compute_expected_log_prob(self):

        for (w, t), val in np.ndenumerate(self.e_log_prob):
            self.e_log_prob[w][t] = self.mean[w][t + 1] - np.log(self.zeta[t])
        return self.e_log_prob

    def state_space_counts_init(self, approx_true_fwd_var, beta_variance, sstats):

        W = self.vocab_len
        T = self.num_articles_per_times

        log_norm_counts = np.copy(sstats)
        log_norm_counts /= sum(log_norm_counts)
        log_norm_counts += 1.0 / W
        log_norm_counts /= sum(log_norm_counts)
        log_norm_counts = np.log(log_norm_counts)

        # setting variational observations to transformed counts
        self.obs = (np.repeat(log_norm_counts, T, axis=0)).reshape(W, T)
        # set variational parameters
        self.approx_true_fwd_var = approx_true_fwd_var
        self.beta_variance = beta_variance

        # compute post variance, mean
        for w in range(W):
            self.variance[w], self.fwd_variance[w] = self.compute_post_variance(w, self.beta_variance)
            self.mean[w], self.fwd_mean[w] = self.compute_post_mean(w, self.beta_variance)

        self.zeta = self.update_zeta()
        self.e_log_prob = self.compute_expected_log_prob()

    def fit_state_space(self, sstats):

        W = self.vocab_len
        bound = 0
        old_bound = 0
        state_space_fit_threshold = 1e-6
        state_space_max_epoch = 2
        converged = state_space_fit_threshold + 1

        # computing variance, fwd_variance
        self.variance, self.fwd_variance = \
            (np.array(x) for x in zip(*(self.compute_post_variance(w, self.beta_variance) for w in range(W))))

        # column sum of sstats
        totals = sstats.sum(axis=0)
        epoch_ = 0


        bound = self.compute_bound(sstats, totals)


        while converged > state_space_fit_threshold and epoch_ < state_space_max_epoch:
            epoch_ += 1
            old_bound = bound
            self.obs, self.zeta = self.update_obs(sstats, totals)

            bound = self.compute_bound(sstats, totals)


            converged = np.fabs((bound - old_bound) / old_bound)

        self.e_log_prob = self.compute_expected_log_prob()
        return bound

    def compute_bound(self, sstats, totals):

        w = self.vocab_len
        t = self.num_articles_per_times

        term_1 = 0
        term_2 = 0
        term_3 = 0

        val = 0
        ent = 0

        beta_variance = self.beta_variance
        # computing mean, fwd_mean
        self.mean, self.fwd_mean = \
            (np.array(x) for x in zip(*(self.compute_post_mean(w, self.beta_variance) for w in range(w))))
        self.zeta = self.update_zeta()

        val = sum(self.variance[w][0] - self.variance[w][t] for w in range(w)) / 2 * beta_variance


        for t in range(1, t + 1):
            term_1 = 0.0
            term_2 = 0.0
            ent = 0.0
            for w in range(w):

                m = self.mean[w][t]
                prev_m = self.mean[w][t - 1]

                v = self.variance[w][t]

                term_1 += \
                    (np.power(m - prev_m, 2) / (2 * beta_variance)) - (v / beta_variance) - np.log(beta_variance)
                term_2 += sstats[w][t - 1] * m
                ent += np.log(v) / 2  # note the 2pi's cancel with term1 (see doc)

            term_3 = -totals[t - 1] * np.log(self.zeta[t - 1])
            val += term_2 + term_3 + ent - term_1

        return val

    def update_obs(self, sstats, totals):


        OBS_NORM_CUTOFF = 2
        STEP_SIZE = 0.01
        TOL = 1e-3

        W = self.vocab_len
        T = self.num_articles_per_times

        runs = 0
        mean_deriv_mtx = np.zeros((T, T + 1))

        norm_cutoff_obs = None
        for w in range(W):
            w_counts = sstats[w]
            counts_norm = 0
            # now we find L2 norm of w_counts
            for i in range(len(w_counts)):
                counts_norm += w_counts[i] * w_counts[i]

            counts_norm = np.sqrt(counts_norm)

            if counts_norm < OBS_NORM_CUTOFF and norm_cutoff_obs is not None:
                obs = self.obs[w]
                norm_cutoff_obs = np.copy(obs)
            else:
                if counts_norm < OBS_NORM_CUTOFF:
                    w_counts = np.zeros(len(w_counts))

                # TODO: apply lambda function
                for t in range(T):
                    mean_deriv_mtx[t] = self.compute_mean_deriv(w, t, mean_deriv_mtx[t])

                deriv = np.zeros(T)
                args = self, w_counts, totals, mean_deriv_mtx, w, deriv
                obs = self.obs[w]

                obs = optimize.fmin_cg(
                    f=f_obs, fprime=df_obs, x0=obs, gtol=TOL, args=args, epsilon=STEP_SIZE, disp=0
                )

                runs += 1

                if counts_norm < OBS_NORM_CUTOFF:
                    norm_cutoff_obs = obs

                self.obs[w] = obs

        self.zeta = self.update_zeta()

        return self.obs, self.zeta

    def compute_mean_deriv(self, word, time, deriv):

        T = self.num_articles_per_times
        fwd_variance = self.variance[word]

        deriv[0] = 0

        # forward pass
        for t in range(1, T + 1):
            if self.approx_true_fwd_var > 0.0:
                w = self.approx_true_fwd_var / (fwd_variance[t - 1] + self.beta_variance + self.approx_true_fwd_var)
            else:
                w = 0.0
            val = w * deriv[t - 1]
            if time == t - 1:
                val += (1 - w)
            deriv[t] = val

        for t in range(T - 1, -1, -1):
            if self.beta_variance == 0.0:
                w = 0.0
            else:
                w = self.beta_variance / (fwd_variance[t] + self.beta_variance)
            deriv[t] = w * deriv[t] + (1 - w) * deriv[t + 1]

        return deriv

    def compute_obs_deriv(self, word, word_counts, totals, mean_deriv_mtx, deriv):

        init_mult = 1000

        T = self.num_articles_per_times

        mean = self.mean[word]
        variance = self.variance[word]

        self.temp_vect = np.zeros(T)

        for u in range(T):
            self.temp_vect[u] = np.exp(mean[u + 1] + variance[u + 1] / 2)

        for t in range(T):
            mean_deriv = mean_deriv_mtx[t]
            term1 = 0
            term2 = 0
            term3 = 0
            term4 = 0

            for u in range(1, T + 1):
                mean_u = mean[u]
                mean_u_prev = mean[u - 1]
                dmean_u = mean_deriv[u]
                dmean_u_prev = mean_deriv[u - 1]

                term1 += (mean_u - mean_u_prev) * (dmean_u - dmean_u_prev)
                term2 += (word_counts[u - 1] - (totals[u - 1] * self.temp_vect[u - 1] / self.zeta[u - 1])) * dmean_u

            if self.beta_variance:
                term1 = - (term1 / self.beta_variance)
                term1 = term1 - (mean[0] * mean_deriv[0]) / (init_mult * self.beta_variance)
            else:
                term1 = 0.0

            deriv[t] = term1 + term2 + term3 + term4

        return deriv

def f_obs(x, *args):

    state_space, word_counts, totals, mean_deriv_mtx, word, deriv = args
    init_mult = 1000

    T = len(x)
    val = 0
    term1 = 0
    term2 = 0

    state_space.obs[word] = x
    state_space.mean[word], state_space.fwd_mean[word] = state_space.compute_post_mean(word, state_space.beta_variance)

    mean = state_space.mean[word]
    variance = state_space.variance[word]

    for t in range(1, T + 1):
        mean_t = mean[t]
        mean_t_prev = mean[t - 1]

        val = mean_t - mean_t_prev
        term1 += val * val
        term2 += word_counts[t - 1] * mean_t - totals[t - 1] * np.exp(mean_t + variance[t] / 2) / state_space.zeta[t - 1]

 

    if state_space.beta_variance > 0.0:

        term1 = - (term1 / (2 * state_space.beta_variance))
        term1 = term1 - mean[0] * mean[0] / (2 * init_mult * state_space.beta_variance)
    else:
        term1 = 0.0

    final = -(term1 + term2)

    return final


def df_obs(x, *args):

    state_space, word_counts, totals, mean_deriv_mtx, word, deriv = args

    state_space.obs[word] = x
    state_space.mean[word], state_space.fwd_mean[word] = state_space.compute_post_mean(word, state_space.beta_variance)

    deriv = state_space.compute_obs_deriv(word, word_counts, totals, mean_deriv_mtx, deriv)

    return np.negative(deriv)
