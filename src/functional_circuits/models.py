"""Small neural models with explicit, finite-difference-tested NumPy gradients."""
import numpy as np

KINDS = ("functional", "monolithic", "random_partition", "dense_modular", "no_gate", "oracle_gate")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def bce(z, y):
    return float(np.mean(np.logaddexp(0, z) - y*z))


class Model:
    def __init__(self, kind, seed=0, width=8):
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind, self.width = kind, width
        self.params = {}
        rng = np.random.default_rng(seed)
        if kind == "monolithic":
            self.groups = [np.arange(13)]
            self._init_head("f", 13, width+1, rng)
        else:
            if kind == "dense_modular":
                self.groups = [np.arange(12), np.arange(12)]
            elif kind == "random_partition":
                # Fixed per training seed, never selected on outcomes.
                permutation = np.random.default_rng(seed+888).permutation(12)
                self.groups = [permutation[:6], permutation[6:]]
            else:
                self.groups = [np.arange(6), np.arange(6, 12)]
            for name, group in zip(("a", "b"), self.groups):
                self._init_head(name, len(group), width, rng)
            if kind not in ("no_gate", "oracle_gate"):
                self.params["gate_w"] = np.zeros(1)
                self.params["gate_b"] = np.zeros(1)

    def _init_head(self, prefix, size, width, rng):
        self.params[prefix+"_W"] = rng.normal(0, 1/np.sqrt(size), size=(size, width))
        self.params[prefix+"_b"] = np.zeros(width)
        self.params[prefix+"_v"] = rng.normal(0, .2, size=width)
        self.params[prefix+"_c"] = np.zeros(1)

    @property
    def parameter_count(self):
        return sum(v.size for v in self.params.values())

    def head(self, prefix, x):
        p = self.params
        h = np.tanh(x @ p[prefix+"_W"] + p[prefix+"_b"])
        return h @ p[prefix+"_v"] + p[prefix+"_c"][0], h

    def gate(self, x):
        if self.kind == "oracle_gate":
            return (x[:, -1]+1)/2
        if self.kind == "no_gate":
            return np.full(len(x), .5)
        return sigmoid(x[:, -1]*self.params["gate_w"][0]+self.params["gate_b"][0])

    def forward(self, x):
        if self.kind == "monolithic":
            z, h = self.head("f", x)
            return z, {"h": h}
        a, ha = self.head("a", x[:, self.groups[0]])
        b, hb = self.head("b", x[:, self.groups[1]])
        g = self.gate(x)
        return (1-g)*a+g*b, {"a": a, "b": b, "ha": ha, "hb": hb, "g": g}

    def loss_grad(self, x, y):
        z, cache = self.forward(x)
        dz = (sigmoid(z)-y)/len(y)
        grads = {}
        def back(prefix, xx, h, delta):
            dh = delta[:, None]*self.params[prefix+"_v"]*(1-h*h)
            grads[prefix+"_W"] = xx.T @ dh
            grads[prefix+"_b"] = dh.sum(axis=0)
            grads[prefix+"_v"] = h.T @ delta
            grads[prefix+"_c"] = np.array([delta.sum()])
        if self.kind == "monolithic":
            back("f", x, cache["h"], dz)
        else:
            g = cache["g"]
            back("a", x[:, self.groups[0]], cache["ha"], dz*(1-g))
            back("b", x[:, self.groups[1]], cache["hb"], dz*g)
            if "gate_w" in self.params:
                dg = dz*(cache["b"]-cache["a"])*g*(1-g)
                grads["gate_w"] = np.array([np.sum(dg*x[:, -1])])
                grads["gate_b"] = np.array([np.sum(dg)])
        return bce(z, y), grads

    def patch(self, base, donor, mapping):
        _, a = self.forward(base)
        _, b = self.forward(donor)
        if self.kind == "monolithic":
            h = a["h"].copy()
            h[:, mapping] = b["h"][:, mapping]
            return h @ self.params["f_v"] + self.params["f_c"][0]
        selected = ("a", "b")[mapping[0]]
        a[selected] = b[selected]
        return (1-a["g"])*a["a"]+a["g"]*a["b"]

    def lesion(self, x, mapping):
        _, a = self.forward(x)
        if self.kind == "monolithic":
            h = a["h"].copy()
            h[:, mapping] = 0
            return h @ self.params["f_v"] + self.params["f_c"][0]
        a[("a", "b")[mapping[0]]] = np.zeros(len(x))
        return (1-a["g"])*a["a"]+a["g"]*a["b"]


def train(model, data, seed, epochs=120, batch_size=128, lr=.01):
    rng = np.random.default_rng(seed)
    first = {k: np.zeros_like(v) for k, v in model.params.items()}
    second = {k: np.zeros_like(v) for k, v in model.params.items()}
    step, losses = 0, []
    for epoch in range(epochs):
        order = rng.permutation(len(data.y))
        for start in range(0, len(order), batch_size):
            batch = order[start:start+batch_size]
            loss, grad = model.loss_grad(data.x[batch], data.y[batch])
            if not np.isfinite(loss) or any(not np.all(np.isfinite(v)) for v in grad.values()):
                raise FloatingPointError("Non-finite training state")
            step += 1
            for k in model.params:
                first[k] = .9*first[k]+.1*grad[k]
                second[k] = .999*second[k]+.001*grad[k]**2
                mh, vh = first[k]/(1-.9**step), second[k]/(1-.999**step)
                model.params[k] -= lr*mh/(np.sqrt(vh)+1e-8)
        if epoch == 0 or (epoch+1) % 10 == 0 or epoch == epochs-1:
            losses.append({"epoch": epoch+1, "loss": bce(model.forward(data.x)[0], data.y)})
    return losses
