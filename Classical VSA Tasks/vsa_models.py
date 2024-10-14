"""
    Implementation heavily influenced by VSATensor and MAPTensor from torchhd:
        https://torchhd.readthedocs.io/en/stable/_modules/torchhd/tensors/base.html#VSATensor
        https://torchhd.readthedocs.io/en/stable/_modules/torchhd/tensors/map.html#MAPTensor
"""
import torch
from torchhd import VSATensor
from torch import Tensor
from typing import List, Set
from math import sqrt


class HLBTensor(VSATensor):
    """ 
    Implements our Hadamard-derived Linear Binding as a VSATensor subclass
    """

    supported_dtypes: Set[torch.dtype] = {
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }

    @classmethod
    def empty(cls,
              num_vectors: int,
              dimensions: int,
              *,
              dtype=None,
              device=None,
              requires_grad=False) -> "HLBTensor":
        """Creates hypervectors representing empty sets"""

        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        result = torch.zeros(
            num_vectors,
            dimensions,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return result.as_subclass(cls)

    @classmethod
    def identity(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "HLBTensor":
        """Creates identity hypervectors for binding"""

        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        result = torch.ones(
            num_vectors,
            dimensions,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return result.as_subclass(cls)

    @classmethod
    def random(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        generator=None,
        requires_grad=False,
    ) -> "HLBTensor":
        """Creates random or uncorrelated hypervectors"""
        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        size = (num_vectors, dimensions)
        uniform = torch.rand(size, generator=generator)
        n1 = torch.normal(-1, 1 / sqrt(dimensions), size, generator=generator)
        n2 = torch.normal(1, 1 / sqrt(dimensions), size, generator=generator)
        result = torch.where(uniform > 0.5, n1, n2).to(device=device)

        result.requires_grad = requires_grad
        return result.as_subclass(cls)

    def bundle(self, other: "HLBTensor") -> "HLBTensor":
        """Bundle the hypervector with other"""

        return torch.add(self, other)

    def multibundle(self) -> "HLBTensor":
        """Bundle multiple hypervectors"""
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} needs to have at least two dimensions for multibundle, got size: {tuple(self.shape)}"
            )

        n = self.size(-2)
        if n == 1:
            return self.unsqueeze(-2)

        tensors: List[VSATensor] = torch.unbind(self, dim=-2)

        output = tensors[0].bundle(tensors[1])
        for i in range(2, n):
            output = output.bundle(tensors[i])

        return output

    def bind(self, other: "HLBTensor") -> "HLBTensor":
        """Bind the hypervector with other"""
        return torch.mul(self, other)

    def multibind(self) -> "HLBTensor":
        """Bind multiple hypervectors"""
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} data needs to have at least two dimensions for multibind, got size: {tuple(self.shape)}"
            )

        n = self.size(-2)
        if n == 1:
            return self.unsqueeze(-2)

        tensors: List[HLBTensor] = torch.unbind(self, dim=-2)

        output = tensors[0].bind(tensors[1])
        for i in range(2, n):
            output = output.bind(tensors[i])

        return output

    def inverse(self) -> "HLBTensor":
        """Inverse the hypervector for binding"""
        return 1 / self

    def negative(self) -> "HLBTensor":
        """Negate the hypervector for the bundling inverse"""
        return torch.negative(self)

    def permute(self, shifts: int = 1) -> "HLBTensor":
        """Permute the hypervector"""
        return torch.roll(self, shifts, dims=-1)

    def dot_similarity(self, others: "HLBTensor", dtype=None) -> Tensor:
        """Inner product with other hypervectors"""
        if dtype is None:
            dtype = torch.get_default_dtype()

        if others.dim() >= 2:
            others = others.transpose(-2, -1)

        return torch.matmul(self.to(dtype), others.to(dtype))

    def cosine_similarity(self, others: "HLBTensor", dtype=None) -> Tensor:
        """Cosine similarity with other hypervectors"""
        if dtype is None:
            dtype = torch.get_default_dtype()

        self_dot = torch.sum(self * self, dim=-1, dtype=dtype)
        self_mag = torch.sqrt(self_dot)

        others_dot = torch.sum(others * others, dim=-1, dtype=dtype)
        others_mag = torch.sqrt(others_dot)

        if self.dim() >= 2:
            magnitude = self_mag.unsqueeze(-1) * others_mag.unsqueeze(-2)
        else:
            magnitude = self_mag * others_mag

        return self.dot_similarity(others, dtype=dtype) / magnitude

    def scaled_cosine_similarity(self,
                                 others: "HLBTensor",
                                 bundle_count: int,
                                 dtype=None) -> Tensor:
        return self.cosine_similarity(others, dtype=dtype) * sqrt(bundle_count)


class MAPCTensor(VSATensor):
    r"""Multiply Add Permute

    Proposed in `Multiplicative Binding, Representation Operators & Analogy <https://www.researchgate.net/publication/215992330_Multiplicative_Binding_Representation_Operators_Analogy>`_, this model works with dense bipolar hypervectors with elements from :math:`\{-1,1\}`.
    """

    supported_dtypes: Set[torch.dtype] = {
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }

    @classmethod
    def empty(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "MAPCTensor":
        r"""Creates a set of hypervectors representing empty sets.

        When bundled with a hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        """

        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        result = torch.zeros(
            num_vectors,
            dimensions,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return result.as_subclass(cls)

    @classmethod
    def identity(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "MAPCTensor":
        r"""Creates a set of identity hypervectors.

        When bound with a random-hypervector :math:`x`, the result is :math:`x`.

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.


        """

        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        result = torch.ones(
            num_vectors,
            dimensions,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return result.as_subclass(cls)

    @classmethod
    def random(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        generator=None,
        dtype=None,
        device=None,
        requires_grad=False,
    ) -> "MAPCTensor":
        r"""Creates a set of random independent hypervectors.

        The resulting hypervectors are sampled uniformly at random from the ``dimensions``-dimensional hyperspace, each component in [-1, 1].

        Args:
            num_vectors (int): the number of hypervectors to generate.
            dimensions (int): the dimensionality of the hypervectors.
            generator (``torch.Generator``, optional): a pseudorandom number generator for sampling.
            dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
            device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

        """
        if dtype is None:
            dtype = torch.get_default_dtype()

        if dtype not in cls.supported_dtypes:
            name = cls.__name__
            options = ", ".join([str(x) for x in cls.supported_dtypes])
            raise ValueError(f"{name} vectors must be one of dtype {options}.")

        size = (num_vectors, dimensions)
        result = torch.empty(size, device=device).uniform_(-1.0,
                                                           1.0,
                                                           generator=generator)
        result.requires_grad = requires_grad
        return result.as_subclass(cls)

    def bundle(self, other: "MAPCTensor") -> "MAPCTensor":
        r"""Bundle the hypervector with other using element-wise sum, clamped to be within [-1, 1]

        This produces a hypervector maximally similar to both.

        The bundling operation is used to aggregate information into a single hypervector.

        Args:
            other (MAPCTensor): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`


        """
        return torch.add(self, other).clamp(-1, 1)

    def multibundle(self) -> "MAPCTensor":
        """Bundle multiple hypervectors"""
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} needs to have at least two dimensions for multibundle, got size: {tuple(self.shape)}"
            )

        n = self.size(-2)
        if n == 1:
            return self.unsqueeze(-2)

        tensors: List[VSATensor] = torch.unbind(self, dim=-2)

        output = tensors[0].bundle(tensors[1])
        for i in range(2, n):
            output = output.bundle(tensors[i])

        return output

    def bind(self, other: "MAPCTensor") -> "MAPCTensor":
        r"""Bind the hypervector with other using element-wise multiplication.

        This produces a hypervector dissimilar to both.

        Binding is used to associate information, for instance, to assign values to variables.

        Args:
            other (MAPTensor): other input hypervector

        Shapes:
            - Self: :math:`(*)`
            - Other: :math:`(*)`
            - Output: :math:`(*)`

        Examples::

            >>> a, b = torchhd.MAPTensor.random(2, 10)
            >>> a
            tensor([ 1., -1.,  1.,  1., -1.,  1.,  1., -1.,  1.,  1.])
            >>> b
            tensor([-1.,  1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1.])
            >>> a.bind(b)
            tensor([-1., -1., -1.,  1., -1.,  1.,  1., -1., -1., -1.])

        """

        return torch.mul(self, other)

    def multibind(self) -> "MAPCTensor":
        """Bind multiple hypervectors"""
        return torch.prod(self, dim=-2, dtype=self.dtype)

    def inverse(self) -> "MAPCTensor":
        r"""Invert the hypervector for binding.

        Each hypervector in MAP is its own inverse, so this returns a copy of self.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        """

        return torch.clone(self)

    def negative(self) -> "MAPCTensor":
        r"""Negate the hypervector for the bundling inverse

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        """

        return torch.negative(self)

    def permute(self, shifts: int = 1) -> "MAPCTensor":
        r"""Permute the hypervector.

        The permutation operator is commonly used to assign an order to hypervectors.

        Args:
            shifts (int, optional): The number of places by which the elements of the tensor are shifted.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        """
        return torch.roll(self, shifts=shifts, dims=-1)

    def clipping(self, kappa) -> "MAPCTensor":
        r"""Performs the clipping function that clips the lower and upper values.

        Args:
            kappa (int): specifies the range of the clipping function.

        Shapes:
            - Self: :math:`(*)`
            - Output: :math:`(*)`

        """

        return torch.clamp(self, min=-kappa, max=kappa)

    def dot_similarity(self, others: "MAPCTensor", *, dtype=None) -> Tensor:
        """Inner product with other hypervectors"""
        if dtype is None:
            dtype = torch.get_default_dtype()

        if others.dim() >= 2:
            others = others.transpose(-2, -1)

        return torch.matmul(self.to(dtype), others.to(dtype))

    def cosine_similarity(self,
                          others: "MAPCTensor",
                          *,
                          dtype=None,
                          eps=1e-08) -> Tensor:
        """Cosine similarity with other hypervectors"""
        if dtype is None:
            dtype = torch.get_default_dtype()

        self_dot = torch.sum(self * self, dim=-1, dtype=dtype)
        self_mag = torch.sqrt(self_dot)

        others_dot = torch.sum(others * others, dim=-1, dtype=dtype)
        others_mag = torch.sqrt(others_dot)

        if self.dim() >= 2:
            magnitude = self_mag.unsqueeze(-1) * others_mag.unsqueeze(-2)
        else:
            magnitude = self_mag * others_mag

        if torch.isclose(magnitude, torch.zeros_like(magnitude),
                         equal_nan=True).any():
            import warnings

            warnings.warn(
                "The norm of a vector is nearly zero, this could indicate a bug."
            )

        magnitude = torch.clamp(magnitude, min=eps)
        return self.dot_similarity(others, dtype=dtype) / magnitude
