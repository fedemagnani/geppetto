use crate::tensor::Tensor;

pub struct SimpleLayer<'a> {
    weight: &'a Tensor,
    bias: &'a Tensor,
}

impl<'a> SimpleLayer<'a> {
    pub fn new(weight: &'a Tensor, bias: &'a Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn eval(&self, input: &Tensor) -> Tensor {
        (input * self.weight) + self.bias
    }
}
