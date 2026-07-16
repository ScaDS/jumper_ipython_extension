import numpy as np

from jumper_extension.adapters.ai_reviewer.benchmark import fingerprint as fp


# --- assigned_names ---

def test_assigned_names_collects_top_level_bindings_in_order():
    code = "import numpy as np\ny = np.sqrt(x).sum()\na, b = 1, 2\ntotal += y"

    assert fp.assigned_names(code) == ["y", "a", "b", "total"]


def test_assigned_names_ignores_item_and_attribute_targets():
    assert fp.assigned_names("d[k] = 1\nobj.attr = 2") == []


def test_assigned_names_survives_broken_code():
    assert fp.assigned_names("def broken(:") == []


# --- tolerance ---

def test_reordered_float_sum_still_counts_as_the_same_result():
    values = np.random.default_rng(0).random(100_000) * 1e6
    sequential = 0.0
    for value in values:
        sequential += value
    pairwise = float(values.sum())

    assert sequential != pairwise  # vectorising reorders the additions
    assert fp.compare(fp.fingerprint(sequential), fp.fingerprint(pairwise)) == fp.MATCH


def test_a_variant_that_processed_fewer_elements_is_caught():
    full = fp.fingerprint(float(np.arange(1000.0).sum()))
    partial = fp.fingerprint(float(np.arange(500.0).sum()))

    assert fp.compare(full, partial) == fp.DIFFERS


def test_arrays_compare_on_shape_and_distribution():
    array = np.sqrt(np.arange(1000.0))

    assert fp.compare(fp.fingerprint(array), fp.fingerprint(np.sqrt(np.arange(1000.0)))) == fp.MATCH
    assert fp.compare(fp.fingerprint(array), fp.fingerprint(np.sqrt(np.arange(500.0)))) == fp.DIFFERS


def test_unsummarisable_value_is_unverified_rather_than_wrong():
    assert fp.compare(fp.fingerprint(object()), fp.fingerprint(object())) == fp.UNVERIFIED


def test_compare_all_reports_which_names_diverged():
    baseline = {"y": fp.fingerprint(10.0), "z": fp.fingerprint(1.0)}
    variant = {"y": fp.fingerprint(99.0), "z": fp.fingerprint(1.0)}

    assert fp.compare_all(baseline, variant) == (fp.DIFFERS, ["y"])


def test_compare_all_is_unverified_when_a_name_is_missing():
    baseline = {"y": fp.fingerprint(10.0)}

    verdict, differing = fp.compare_all(baseline, {})

    assert verdict == fp.UNVERIFIED
    assert differing == []
