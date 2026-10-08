"""Compact GA with Deb constraint ordering, identity fallback and MIDI cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np

from ..midi.types import InternalRepr
from .objective import E3Objective, ranking_key
from .profile import TargetProfile
from .transformation import apply_transformation
from .types import CandidateEvaluation, E3Genome, IDENTITY_GENOME


@dataclass(frozen=True)
class SearchConfig:
    population_size: int = 32
    generations: int = 60
    elitism_k: int = 2
    tournament_size: int = 3
    stagnation_generations: int = 12
    mutation_sigma: tuple[float, float, float, float, float] = (1.0, 0.12, 0.12, 0.12, 0.12)

    def __post_init__(self) -> None:
        if self.population_size < 2 or self.generations < 0:
            raise ValueError("invalid population_size/generations")
        if not 1 <= self.elitism_k <= self.population_size or self.tournament_size < 1:
            raise ValueError("invalid elitism/tournament size")
        if self.stagnation_generations < 1 or any(value < 0 for value in self.mutation_sigma):
            raise ValueError("invalid stagnation/mutation sigma")


@dataclass(frozen=True)
class SearchResult:
    genome: E3Genome
    output: InternalRepr
    evaluation: CandidateEvaluation
    history: tuple[dict[str, object], ...]
    stop_reason: str
    unique_candidates: int
    cache_hits: int


def _clamp(genome: E3Genome) -> E3Genome:
    return E3Genome(
        transpose_semitones=int(min(6, max(-6, round(genome.transpose_semitones)))),
        rhythm_strength=float(min(1, max(0, genome.rhythm_strength))),
        duration_strength=float(min(1, max(0, genome.duration_strength))),
        thin_strength=float(min(1, max(0, genome.thin_strength))),
        double_strength=float(min(1, max(0, genome.double_strength))),
    )


class E3GeneticAlgorithm:
    def __init__(self, config: SearchConfig = SearchConfig()) -> None:
        self.config = config

    def run(
        self,
        source: InternalRepr,
        profile: TargetProfile,
        *,
        seed: int,
        objective: E3Objective | None = None,
        progress: Callable[[dict[str, object]], None] | None = None,
        canonicalize: Callable[[E3Genome], E3Genome] | None = None,
        transform: Callable[..., InternalRepr] | None = None,
    ) -> SearchResult:
        """Run the search with optional genome and transformation callbacks.

        ``canonicalize`` replaces the default clamp before cache lookup and
        when creating offspring. ``transform`` receives ``source``, ``genome``,
        ``profile``, and keyword arguments ``seed`` and ``structure``; it runs
        only on genome-cache misses. Callbacks must be deterministic for caching.
        """
        if canonicalize is None:
            canonicalize = _clamp
        if transform is None:
            transform = apply_transformation
        rng = np.random.default_rng(seed)
        objective = objective or E3Objective(source, profile)
        population = [IDENTITY_GENOME]
        while len(population) < self.config.population_size:
            population.append(E3Genome(
                int(rng.integers(-6, 7)), float(rng.random()), float(rng.random()),
                float(rng.random()), float(rng.random()),
            ))
        output_cache: dict[InternalRepr, CandidateEvaluation] = {}
        genome_cache: dict[E3Genome, CandidateEvaluation] = {}
        cache_hits = 0

        def evaluate(genome: E3Genome) -> CandidateEvaluation:
            nonlocal cache_hits
            genome = canonicalize(genome)
            if genome in genome_cache:
                cache_hits += 1
                return genome_cache[genome]
            output = transform(source, genome, profile, seed=seed, structure=objective.structure)
            if output in output_cache:
                cache_hits += 1
                cached = output_cache[output]
                value = replace(cached, genome=genome)
            else:
                value = objective.evaluate(genome, output)
                output_cache[output] = value
            genome_cache[genome] = value
            return value

        identity = evaluate(IDENTITY_GENOME)
        history: list[dict[str, object]] = []
        best = identity
        stagnant = 0
        stop_reason = "max_generations"
        for generation in range(self.config.generations + 1):
            evaluated = [evaluate(genome) for genome in population]
            evaluated.sort(key=ranking_key, reverse=True)
            current = evaluated[0]
            if ranking_key(current) > ranking_key(best):
                best, stagnant = current, 0
            else:
                stagnant += 1 if generation else 0
            feasible_count = sum(item.constraints.feasible for item in evaluated)
            row: dict[str, object] = {
                "generation": generation,
                "best_style_gain": current.style_gain,
                "best_feasible": current.constraints.feasible,
                "feasible_count": feasible_count,
                "unique_candidates": len(output_cache),
                "cache_hits": cache_hits,
                "best_genome": asdict(current.genome),
            }
            history.append(row)
            if progress:
                progress(row)
            if generation == self.config.generations:
                break
            if stagnant >= self.config.stagnation_generations:
                stop_reason = "stagnation"
                break
            elites = [item.genome for item in evaluated[: self.config.elitism_k]]
            def tournament() -> E3Genome:
                indices = rng.integers(0, len(evaluated), size=self.config.tournament_size)
                return max((evaluated[int(index)] for index in indices), key=ranking_key).genome

            next_population = list(elites)
            while len(next_population) < self.config.population_size:
                left, right = tournament(), tournament()
                mask = rng.random(5) < 0.5
                left_values = list(asdict(left).values())
                right_values = list(asdict(right).values())
                values = [left_values[i] if mask[i] else right_values[i] for i in range(5)]
                values = [value + float(rng.normal(0, sigma)) for value, sigma in zip(values, self.config.mutation_sigma)]
                next_population.append(canonicalize(E3Genome(*values)))
            population = next_population

        # Identity is the explicit safe fallback, including when all candidates
        # violate a hard constraint or the best feasible style gain is negative.
        if not best.constraints.feasible or best.style_gain < 0.0:
            best = identity
            stop_reason += "_identity_fallback"
        return SearchResult(
            best.genome, best.output, best, tuple(history), stop_reason,
            len(output_cache), cache_hits,
        )
