import unittest
import numpy as np
from functional_circuits.data import generate, retrieve, evidence_score, memory_counterfactual
from functional_circuits.models import Model, KINDS, train, bce
from functional_circuits.metrics import accuracy, choose_mapping, evaluate


class DataTests(unittest.TestCase):
    def test_generation_and_labels(self):
        a = generate(2000, 15, .8)
        np.testing.assert_array_equal(a.sensory_y, evidence_score(a.x[:, :6], "sensory") > 0)
        np.testing.assert_array_equal(a.memory_y, evidence_score(a.x[:, 6:12], "memory") > 0)
        np.testing.assert_array_equal(a.y, np.where(a.context, a.memory_y, a.sensory_y))
        self.assertAlmostEqual(np.mean(a.sensory_y == a.memory_y), .8, delta=.04)

    def test_determinism_and_separation(self):
        self.assertEqual(generate(30, 1).fingerprint(), generate(30, 1).fingerprint())
        self.assertNotEqual(generate(30, 1).fingerprint(), generate(30, 2).fingerprint())

    def test_exact_retrieval(self):
        bank = np.arange(24).reshape(3, 4, 2)
        got = retrieve(bank, np.array([2, 0, 3]))
        np.testing.assert_array_equal(got, np.array([[4, 5], [8, 9], [22, 23]]))
        got[0] = -1
        self.assertEqual(bank[0, 2, 0], 4)

    def test_extreme_agreement(self):
        for agreement in (0, 1):
            d = generate(100, 8, agreement)
            self.assertTrue(np.all((d.sensory_y == d.memory_y) == agreement))

    def test_counterfactual(self):
        a, b = generate(100, 1), generate(100, 2)
        target = memory_counterfactual(a, b)
        np.testing.assert_array_equal(target[a.context == 0], a.y[a.context == 0])
        np.testing.assert_array_equal(target[a.context == 1], b.memory_y[a.context == 1])

    def test_invalid_generation(self):
        with self.assertRaises(ValueError): generate(0, 1)
        with self.assertRaises(ValueError): generate(1, 1, 1.2)


class ModelTests(unittest.TestCase):
    def test_gradients_every_parameter(self):
        d = generate(7, 99, .5)
        eps = 1e-5
        for kind in KINDS:
            model = Model(kind, 9, width=3)
            _, grads = model.loss_grad(d.x, d.y)
            for key, parameter in model.params.items():
                for index in np.ndindex(parameter.shape):
                    old = parameter[index]
                    parameter[index] = old+eps
                    upper = bce(model.forward(d.x)[0], d.y)
                    parameter[index] = old-eps
                    lower = bce(model.forward(d.x)[0], d.y)
                    parameter[index] = old
                    self.assertAlmostEqual((upper-lower)/(2*eps), grads[key][index], delta=2e-7,
                                           msg=f"{kind}: {key} {index}")

    def test_parameter_counts(self):
        self.assertEqual(Model("functional").parameter_count, 132)
        self.assertEqual(Model("random_partition").parameter_count, 132)
        self.assertEqual(Model("monolithic").parameter_count, 136)
        self.assertEqual(Model("dense_modular").parameter_count, 228)
        self.assertEqual(Model("no_gate").parameter_count, 130)

    def test_self_patch_is_identity(self):
        d = generate(30, 10)
        for kind in KINDS:
            m = Model(kind, 12)
            mapping = [0, 3] if kind == "monolithic" else [1]
            np.testing.assert_allclose(m.forward(d.x)[0], m.patch(d.x, d.x, mapping), atol=1e-14)

    def test_role_restriction(self):
        d, donor = generate(30, 10), generate(30, 11)
        m = Model("functional", 12)
        modified = d.x.copy()
        modified[:, 6:12] = donor.x[:, 6:12]
        np.testing.assert_array_equal(m.forward(d.x)[1]["a"], m.forward(modified)[1]["a"])
        modified = d.x.copy()
        modified[:, :6] = donor.x[:, :6]
        np.testing.assert_array_equal(m.forward(d.x)[1]["b"], m.forward(modified)[1]["b"])

    def test_training_is_deterministic_and_learns(self):
        d = generate(512, 71, .5)
        a, b = Model("functional", 3), Model("functional", 3)
        la = train(a, d, 44, epochs=50)
        lb = train(b, d, 44, epochs=50)
        self.assertEqual(la, lb)
        for k in a.params: np.testing.assert_array_equal(a.params[k], b.params[k])
        self.assertGreater(accuracy(a.forward(d.x)[0], d.y), .9)
        self.assertLess(la[-1]["loss"], la[0]["loss"])

    def test_mapping_budget_and_evaluation(self):
        a, b = generate(80, 17, .5), generate(80, 19, .5)
        for kind in ("functional", "monolithic"):
            m = Model(kind, 8)
            mapping, metadata = choose_mapping(m, a, b)
            expected = 45 if kind == "monolithic" else 2
            self.assertEqual(metadata["mapping_candidates"], expected)
            result = evaluate(m, a, b, mapping)
            self.assertTrue(0 <= result["memory_iia"] <= 1)
            self.assertTrue(0 <= result["memory_iia_changed"] <= 1)


if __name__ == "__main__":
    unittest.main()
