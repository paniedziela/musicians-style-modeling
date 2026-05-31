"""Parser_MIDI, Pretty_Printer_MIDI i konwerter Pianoroll (Wymagania 5.x).

Pakiet udostępnia komponenty spójnego, dwukierunkowego potoku MIDI:

``parse → Pianoroll.from_internal → [Model_GAN] → Pianoroll.to_internal → write``

* :class:`MidiParser` - plik SMF → :class:`InternalRepr` (lista zdarzeń).
* :class:`Pianoroll` - :class:`InternalRepr` ↔ binarna macierz ``[T × P]``
  (pomocnicza reprezentacja *Modelu_GAN*; konwersja w stronę macierzy jest
  stratna, rekonstrukcja korzysta z *template*, Wymaganie 5.7).
* :class:`MidiPrettyPrinter` - :class:`InternalRepr` → plik SMF.
"""

from .parser import MidiParser
from .pianoroll import Pianoroll
from .printer import MidiPrettyPrinter
from .types import InternalRepr, MetaEvent, NoteEvent, event_key

__all__ = [
    "MidiParser",
    "MidiPrettyPrinter",
    "Pianoroll",
    "InternalRepr",
    "NoteEvent",
    "MetaEvent",
    "event_key",
]
