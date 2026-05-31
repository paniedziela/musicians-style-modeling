"""Pipeline transferu stylu (inferencja GAN / GA / łączona) (Wymagania 5.x).

Pakiet udostępnia :class:`StyleTransferPipeline` - *Potok inferencji* z sekcji
*Pipeline inferencji* dokumentu ``design.md`` (zadania 10.1-10.3):

* :meth:`~musicians_style.inference.pipeline.StyleTransferPipeline.infer_gan` -
  inferencja generatywna *Modelu_GAN* z walidacją ``target_artist`` względem
  metadanych *Punktu_Kontrolnego* (Wymagania 5.1, 5.2, 5.6-5.10),
* :meth:`~musicians_style.inference.pipeline.StyleTransferPipeline.infer_ga` -
  inferencja ewolucyjna na cechach *Zbioru_Stylu* (Wymagania 5.1, 5.11),
* :meth:`~musicians_style.inference.pipeline.StyleTransferPipeline.infer_combined`
  - tryb łączony GAN + GA (Wymaganie 5.1).
"""

from .pipeline import CombinedOrder, StyleTransferPipeline

__all__ = ["StyleTransferPipeline", "CombinedOrder"]
