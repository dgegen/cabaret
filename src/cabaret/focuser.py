from dataclasses import dataclass


@dataclass
class Focuser:
    """A simple focuser model.

    Parameters
    ----------
    position : float
        The current focus position.
    best_position : float
        The best focus position.
    scale : float
        The scale factor for defocus.
    max_seeing_multiplier : float
        The maximum seeing multiplier possible due to defocus.

    Examples
    --------
    >>> from cabaret.focuser import Focuser
    >>> focuser = Focuser()
    >>> focuser.seeing_multiplier(focus_position=10100)
    2.0
    """

    position: float = 10_000
    best_position: float = 10_000
    scale: float = 100
    max_seeing_multiplier: float = 5.0

    @property
    def seeing_multiplier(
        self,
    ) -> float:
        """Factor by which the seeing is increased due to defocus."""
        offset = abs(self.position - self.best_position)
        return min(1 + offset / self.scale, self.max_seeing_multiplier)
