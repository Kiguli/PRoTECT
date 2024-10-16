#Edited by Ben Wooding for purpose of comparisons with PRoTECT

from experiments.benchmarks import models
from fossil import domains
from fossil import certificate
from fossil import main, control
from fossil.consts import *


def test_lnn(args):
    batch_size = 3000
    f = models.HighOrd6B
    n_vars = f.n_vars

    XD = domains.Rectangle([-2] * n_vars, [2] * n_vars)
    # XD = domains.Sphere([0] * n_vars, 2)
    XI = domains.Rectangle([0.5] * n_vars, [1.5] * n_vars)
    # XI = domains.Sphere([1] * n_vars, 0.1)
    XU = domains.Rectangle([-2.4] * n_vars, [-1.6] * n_vars)
    # XU = domains.Sphere([-2] * n_vars, 0.2)
    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        # certificate.XS_BORDER: XS,
        certificate.XU: XU,
        # certificate.XG: XG,
    }
    data = {
        certificate.XD: XD._generate_data(batch_size),
        certificate.XI: XI._generate_data(batch_size),
        certificate.XU: XU._generate_data(batch_size),
    }

    # define NN parameters
    activations = [ActivationType.LINEAR]
    n_hidden_neurons = [10] * len(activations)

    opts = CegisConfig(
        DOMAINS=sets,
        DATA=data,
        SYSTEM=f,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.Z3,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=1000,
        SYMMETRIC_BELT=True,
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
