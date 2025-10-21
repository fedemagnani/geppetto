use super::*;

pub struct Dataset {
    pub train: TensorDataset,
    pub test: TensorDataset,
}

impl Dataset {
    pub fn new(train: TensorDataset, test: TensorDataset) -> Self {
        Self { train, test }
    }

    pub fn from_path<P: AsRef<Path>>(
        path: P,
        tokenizer: &Tokenizer,
        device: &Device,
        train_pct: f32,
    ) -> eyre::Result<Self> {
        let values = fs::read_to_string(path)?;

        //map the whole text in tokenids
        let token_ids = tokenizer.as_ref().encode_with_special_tokens(&values);

        let max_length = 4;
        let stride = 2;

        // create the input-target matrices
        let mut inputs = vec![];
        let mut targets = vec![];

        for window in token_ids.windows(max_length + 1).step_by(stride) {
            let input = window[..max_length].to_vec();
            let target = window[1..].to_vec();

            inputs.push(input);
            targets.push(target);
        }

        let split_i = train_pct * inputs.len() as f32;
        let split_i = split_i.round() as usize;

        // Split into train/test slices
        let (train_inputs, test_inputs) = inputs.split_at(split_i);
        let (train_targets, test_targets) = targets.split_at(split_i);

        let train = TensorDataset::from_vec(train_inputs.to_vec(), train_targets.to_vec(), device)?;
        let test = TensorDataset::from_vec(test_inputs.to_vec(), test_targets.to_vec(), device)?;

        Ok(Self::new(train, test))
    }
}

pub struct TensorDataset {
    pub inputs: Tensor,
    pub targets: Tensor,
}

impl TensorDataset {
    pub fn new(inputs: Tensor, targets: Tensor) -> Self {
        Self { inputs, targets }
    }

    pub fn from_vec(
        inputs: Vec<Vec<u32>>,
        targets: Vec<Vec<u32>>,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let n = inputs.len();
        if n == 0 {
            return Ok(Self {
                inputs: Tensor::zeros(0, DType::U32, device)?,
                targets: Tensor::zeros(0, DType::U32, device)?,
            });
        }
        let emb_dim = inputs[0].len();
        let inputs = inputs.into_iter().flatten();
        let targets = targets.into_iter().flatten();
        let inputs = Tensor::from_iter(inputs, device)?.reshape((n, emb_dim))?;
        let targets = Tensor::from_iter(targets, device)?.reshape((n, emb_dim))?;

        Ok(Self { inputs, targets })
    }

    pub fn inputs(&self) -> &Tensor {
        &self.inputs
    }

    pub fn targets(&self) -> &Tensor {
        &self.targets
    }

    pub fn batches(
        &self,
        batch_size: usize,
        shuffle: bool,
    ) -> impl Iterator<Item = candle_core::Result<(Tensor, Tensor)>> {
        let n = self.inputs.dims()[0];
        let mut start_indices: Vec<usize> = (0..n - batch_size).step_by(batch_size).collect();
        if shuffle {
            start_indices.shuffle(&mut rng());
        }

        start_indices.into_iter().map(move |start| {
            let end = (start + batch_size).min(n);
            Ok((
                self.inputs.narrow(0, start, end - start)?,
                self.targets.narrow(0, start, end - start)?,
            ))
        })
    }
}
