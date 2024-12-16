from .dual import Dual
from .dual_gated_rnn import Dual as Dual_Gated
from .dual_zero_init_atten import Dual as Dual_Zero_Atten
from .dual_zero_nov_att import Dual as Dual_Zero_nov_att
from .mat_rnn import MatRNN
from .mat import MATnet
from .dual_att_momentum import Dual as Dual_mom

MODELS2CLS = {
    'dual': Dual,
    'dual_gate': Dual_Gated,
    'dual_zero_atten': Dual_Zero_Atten,
    'mat': MATnet,
    'matrnn': MatRNN,
    'dual_zero_nov_att': Dual_Zero_nov_att,
    'dual_mom': Dual_mom
}


def build_model(embedding, args):
    cls = MODELS2CLS[args.arch]
    if args.backbone == "clip":
        feature_dim = 512
    if args.arch in ['dual_zero', 'dual_zero_atten','dual_zero_nov', 'dual_zero_nov_att']:
        model = cls(embedding, args.v_feature_dropout_prob, args.dropout, 
        scale=args.sum_scale, reduce_method=args.reduce_method, feature_dim=feature_dim)
    elif args.arch == 'dual_mom':
        print("============build momentum model==========ratio",args.momentum_m)
        model = cls(embedding, args.v_feature_dropout_prob, args.dropout, scale=args.sum_scale,
                     reduce_method=args.reduce_method, momentum=args.momentum_m, feature_dim=feature_dim)
    else:
        model = cls(embedding, args.v_feature_dropout_prob, args.dropout)
    return model
