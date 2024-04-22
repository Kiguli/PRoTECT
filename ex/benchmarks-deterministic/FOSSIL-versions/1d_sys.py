#Edited by Ben Wooding for purpose of comparisons with PRoTECT

from experiments.benchmarks import models
from fossil import domains
from fossil import certificate
from fossil import main
from fossil.consts import *


def test_lnn(args):
    batch_size = 5000
    open_loop = models.RoomTemp1d
    n_vars = open_loop.n_vars

    XD = domains.Rectangle(lb=[-6], ub=[6])
    XI = domains.Rectangle(lb=[-0.5], ub=[0.5])
    XU = domains.Rectangle(lb=[-6], ub=[-5])
    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    data = {
        certificate.XD: XD._generate_data(batch_size),
        certificate.XI: XI._generate_data(batch_size),
        certificate.XU: XU._generate_data(batch_size),
    }

    # define NN parameters
    barr_activations = [ActivationType.SQUARE]
    barr_hidden_neurons = [2] * len(barr_activations)

    opts = CegisConfig(
        SYSTEM=open_loop,
        DOMAINS=sets,
        DATA=data,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.BARRIERALT, #BARRIERALT
        TIME_DOMAIN=TimeDomain.DISCRETE,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=barr_activations,
        N_HIDDEN_NEURONS=barr_hidden_neurons,
        SYMMETRIC_BELT=False,
        CEGIS_MAX_ITERS=50,
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
