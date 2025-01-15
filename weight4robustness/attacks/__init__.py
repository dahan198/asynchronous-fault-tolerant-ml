from .alie import ALittleIsEnoughAttack
from .lf import LabelFlippingAttack
from .sf import SignFlippingAttack
from .ipm import IPMAttack
from ..utils import filter_valid_args

# Define a registry mapping identifiers to classes
ATTACK_REGISTRY = {
    'sf': SignFlippingAttack,
    'lf': LabelFlippingAttack,
    'ipm': IPMAttack,
    'alie': ALittleIsEnoughAttack
}


def get_attack(attack_type, **kwargs):
    """
    Initialize and return an attack class based on the attack_type.
    Filters kwargs to only include relevant initialization parameters for the attack class.

    :param attack_type: A string identifier for the attack type.
    :param kwargs: Keyword arguments for the attack class initializer.
    :return: An instance of the specified attack class.
    """
    if attack_type not in ATTACK_REGISTRY:
        raise ValueError(f"Attack type '{attack_type}' is not supported.")

    attack_class = ATTACK_REGISTRY[attack_type]

    # Filter kwargs based on the constructor signature of the attack class
    filtered_kwargs = filter_valid_args(attack_class, **kwargs)

    attack_instance = attack_class(**filtered_kwargs)

    return attack_instance
