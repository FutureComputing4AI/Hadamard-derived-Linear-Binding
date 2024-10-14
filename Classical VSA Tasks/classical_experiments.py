"""
Experiments used in the classical vsa section
"""
import torch
from torch.nn import CosineSimilarity
import scipy.stats as st
import torchhd
import numpy as np

from vsa_models import HLBTensor, MAPCTensor

from math import sqrt
from collections import defaultdict
import argparse
import json
import os

UPPER = 'upper'
LOWER = 'lower'
AVG = 'avg'


def repeat_bind_magnitude_tests(vsa_classes,
                                max_terms: int,
                                trials: int,
                                dim: int,
                                auto_bind: bool = False):
    """
    Measure the magnitude of hypervectors from various VSAs 
    after repeated bindings
    """
    results = defaultdict(dict)

    for Vsa_Class in vsa_classes:
        class_name = class_obj_to_class_name(Vsa_Class)

        class_results = []
        for _ in range(trials):
            trial_results = []

            for num_bound in range(1, max_terms + 1):

                vectors = None
                if auto_bind:
                    vectors = Vsa_Class.random(1, dim).repeat(num_bound, 1)
                else:
                    vectors = Vsa_Class.random(num_bound, dim)

                full_binding = vectors.multibind().squeeze()

                result = torch.norm(full_binding)
                trial_results.append(result)
            class_results.append(trial_results)

        class_results = np.array(class_results)
        results[class_name][LOWER] = []
        results[class_name][UPPER] = []
        results[class_name][AVG] = []
        for bind_amt in range(class_results.shape[1]):
            avg = float(np.mean(class_results[:, bind_amt]))
            lower = avg
            upper = avg
            std_err = st.sem(class_results[:, bind_amt])
            if std_err != 0:
                lower, upper = st.norm.interval(.95, loc=avg, scale=std_err)
            results[class_name][LOWER].append(lower)
            results[class_name][UPPER].append(upper)
            results[class_name][AVG].append(avg)
    return results


def repeat_bind_cos_sim_tests(vsa_classes,
                              max_terms: int,
                              trials: int,
                              dim: int,
                              auto_bind: bool = False):
    """
    Repeatedly bind, then unbind hypervectors of various VSAs 
    and compute the cosine similarity with the original vector.
    """

    results = defaultdict(dict)

    for Vsa_Class in vsa_classes:
        class_name = class_obj_to_class_name(Vsa_Class)

        class_results = []
        for _ in range(trials):
            trial_results = []

            initial_vector: torchhd.VSATensor = None
            binding: torchhd.VSATensor = None
            bound_terms = []
            for num_bound in range(1, max_terms + 1):

                if num_bound == 1:
                    initial_vector = Vsa_Class.random(1, dim).squeeze()
                    binding = initial_vector
                else:
                    if auto_bind:
                        binding = binding.bind(initial_vector)
                        bound_terms.append(initial_vector)
                    else:
                        new_vec = Vsa_Class.random(1, dim).squeeze()
                        binding = binding.bind(new_vec)
                        bound_terms.append(new_vec)

                # unbind everything but the original vector
                retrieved = binding
                for term in reversed(bound_terms):
                    inv = term.inverse()
                    retrieved = retrieved.bind(inv)

                cos_sim = initial_vector.cosine_similarity(retrieved)
                trial_results.append(cos_sim)
            class_results.append(trial_results)

        class_results = np.array(class_results)
        results[class_name][LOWER] = []
        results[class_name][UPPER] = []
        results[class_name][AVG] = []
        for bind_amt in range(class_results.shape[1]):
            avg = float(np.mean(class_results[:, bind_amt]))
            lower = avg
            upper = avg
            std_err = st.sem(class_results[:, bind_amt])
            if std_err != 0:
                lower, upper = st.norm.interval(.95, loc=avg, scale=std_err)
            results[class_name][LOWER].append(lower)
            results[class_name][UPPER].append(upper)
            results[class_name][AVG].append(avg)
    return results


def retrieval_acc_by_dim(Vsa_Class: torchhd.VSATensor, dims: list,
                         max_terms: int, pool_size: int, trials: int):
    """
    Determine how accurately a VSA can retrieve a particular vector from
    a bundle with varying number of components over varying dimensions.
    """
    results = defaultdict(dict)

    for dim in dims:

        dim_results = []
        for _ in range(trials):
            trial_results = []
            pool = Vsa_Class.random(pool_size, dim)

            left_hand, right_hand = draw_pairs(pool, max_terms)
            bound_terms = left_hand.bind(right_hand)
            binding: Vsa_Class = None

            for num_bound in range(1, max_terms + 1):
                correct = 0
                new_term = bound_terms[num_bound - 1]
                if num_bound == 1:
                    binding: Vsa_Class = new_term
                else:
                    binding: Vsa_Class = binding.bundle(new_term)

                # try to retrieve the hypervectors in the bundle
                for i in range(num_bound):
                    target_vector = left_hand[i]
                    key_vector = right_hand[i]
                    retrieved = binding.bind(key_vector.inverse())

                    best_guess = cos_sim_argmax(retrieved, pool, num_bound)
                    if all(torch.eq(best_guess, target_vector)):
                        correct += 1
                trial_results.append(correct / num_bound)
            dim_results.append(trial_results)

        dim_results = np.array(dim_results)
        results[dim][UPPER] = []
        results[dim][LOWER] = []
        results[dim][AVG] = []
        for bind_amt in range(dim_results.shape[1]):
            avg = float(np.mean(dim_results[:, bind_amt]))
            lower = avg
            upper = avg
            std_err = st.sem(dim_results[:, bind_amt])
            if std_err != 0:
                lower, upper = st.norm.interval(.95, loc=avg, scale=std_err)
            results[dim][UPPER].append(upper)
            results[dim][LOWER].append(lower)
            results[dim][AVG].append(avg)

    return results


def draw_pairs(pool: torchhd.VSATensor, num_pairs: int):
    """
    Draw `num_pairs` pairs of hypervectors from `pool`

    Returns two VsaTensors which each contain `num_pairs` different hypervectors
    """
    pool_size = pool.shape[0]
    idxs = torch.randint(0, pool_size - 1, (2, num_pairs))
    return pool[idxs[0]], pool[idxs[1]]


def cos_sim_argmax(retrieved: torchhd.VSATensor, pool: torchhd.VSATensor,
                   num_bound):
    """
    Return argmax_{z in pool} cos(retrieved, z)
    
    Cosine similarity is scaled by sqrt(num_bound) if `retrieved` is an instance of HLBTensor
    """
    scale_factor = 1
    if isinstance(retrieved, HLBTensor):
        scale_factor = sqrt(num_bound)
    cos = CosineSimilarity()
    cos_sims = cos(pool, retrieved) * scale_factor
    return pool[torch.argmax(cos_sims)]


def class_obj_to_class_name(class_obj):
    """Convenience function to create human readable dict keys"""
    stringy = str(class_obj)
    return stringy[stringy.rfind('.') + 1:stringy.rfind("'")]


def paper_retrieval_acc(target_dir):
    vsa_classes = [HLBTensor, torchhd.VTBTensor, torchhd.HRRTensor, MAPCTensor]
    vsa_names = ["HLB", "VTB", "HRR", "MAP-C"]
    retrieval_dims = (144, 256, 400, 576, 784, 1024, 1296)
    max_terms = 25
    pool_size = 1000
    retrieval_trials = 50

    for i in range(len(vsa_classes)):
        results = retrieval_acc_by_dim(vsa_classes[i], retrieval_dims, max_terms, pool_size, retrieval_trials)

        with open(target_dir + f"{vsa_names[i]}_{pool_size}pool_{max_terms}terms_{retrieval_trials}trials.json",
                  'w') as f:
            json.dump(results, f)


def paper_binding(target_dir, magnitude_test: bool, auto=False):
    vsa_classes = [HLBTensor, torchhd.VTBTensor, torchhd.HRRTensor, MAPCTensor]
    max_terms = 25
    binding_trials = 100
    d = 2025

    bind_type = "auto" if auto else "rand"
    results = None
    measure = ""
    if magnitude_test:
        results = repeat_bind_magnitude_tests(vsa_classes, max_terms, binding_trials, dim=d, auto_bind=auto)
        measure = "magnitude"
    else:
        results = repeat_bind_cos_sim_tests(vsa_classes, max_terms, binding_trials, dim=d, auto_bind=auto)
        measure = "cos_sim"

    with open(target_dir + f"{d}d_{bind_type}_{binding_trials}trials_{measure}.json", 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    target_dir = "vsa_results"

    parser = argparse.ArgumentParser(
        description="Run the HLB Classical VSA Tasks Experiments"
    )
    # still need to add help description for each
    parser.add_argument("--all", "-a", action="store_true", help="Perform all experiments from classical VSA section")
    parser.add_argument("--retrieval", action="store_true", help="Perform the retrieval accuracy experiment")
    parser.add_argument("--magnitude", action="store_true", help="Measure the magnitude of bound vectors")
    parser.add_argument("--similarity", action="store_true",
                        help="Measure the cosine similarity after binding and unbinding")
    parser.add_argument("--auto", action="store_true",
                        help="Repeatedly bind the same vector in the magnitude and similarity experiments")
    parser.add_argument("--target", default=target_dir,
                        help=f"Directory to save the output JSON in. Defaults to {target_dir}, will create a directory if it does not exist")

    args = parser.parse_args()

    target_dir = args.target
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    if target_dir[-1] != "/":
        target_dir += "/"

        # retrieval accuracy test
    if args.all or args.retrieval:
        paper_retrieval_acc(target_dir)

    if args.all or args.magnitude:
        if args.all:
            paper_binding(target_dir, magnitude_test=True, auto=True)
            paper_binding(target_dir, magnitude_test=True, auto=False)
        else:
            paper_binding(target_dir, magnitude_test=True, auto=args.auto)

    if args.all or args.similarity:
        if args.all:
            paper_binding(target_dir, magnitude_test=False, auto=True)
            paper_binding(target_dir, magnitude_test=False, auto=False)
        else:
            paper_binding(target_dir, magnitude_test=False, auto=args.auto)
