#Edited by Ben Wooding for purpose of comparisons with PRoTECT

from fossil import domains
from fossil import certificate
from fossil import main
from experiments.benchmarks import models
from fossil.consts import *


def test_lnn(args):
    system = models.Jet2
    XD = domains.Rectangle([0.1, 0.1], [1, 1])
    XI =  domains.Rectangle([0.1, 0.1], [0.5, 0.5])
    XU = domains.Rectangle([0.7, 0.7], [1, 1])

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    data = {
        certificate.XD: XD._generate_data(1000),
        certificate.XI: XI._generate_data(400),
        certificate.XU: XU._generate_data(400),
    }

    # define NN parameters
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    n_hidden_neurons = [10] * len(activations)

    opts = CegisConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=False,
        CEGIS_MAX_ITERS=25,
        VERBOSE=0,
    )

    main.run_benchmark(
        opts,
        record=args.record,
        plot=args.plot,
        concurrent=args.concurrent,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
