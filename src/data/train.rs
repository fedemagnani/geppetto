use super::*;

pub struct Trainer {
    pub entropy: CrossEntropy,
    pub optimizer: AdamW,
    pub shuffle_batches: bool,
}

impl Trainer {
    pub fn new(entropy: CrossEntropy, optimizer: AdamW, shuffle_batches: bool) -> Self {
        Self {
            entropy,
            optimizer,
            shuffle_batches,
        }
    }
    pub fn train(
        &mut self,
        model: &GPTModel,
        data: Dataset,
        epochs: usize,
        batch_size: usize,
    ) -> eyre::Result<()> {
        let mut i = 0;
        for epoch in 0..epochs {
            for batch in data.train.batches(batch_size, self.shuffle_batches) {
                let (input_batch, target_batch) = batch?;
                let loss = self
                    .entropy
                    .loss_tensor(model, &input_batch, &target_batch)?;
                self.optimizer.backward_step(&loss)?;

                if i % 10 == 0 {
                    println!(
                        "Epoch: {epoch}/{epochs}, Batch: {i}, Train Loss: {}",
                        loss.to_scalar::<f32>()?
                    );
                    io::stdout().flush()?;

                    // --- Validation Loop ---
                    let mut val_loss_sum = 0.0;
                    let mut val_batches = 0;
                    for batch in data.test.batches(batch_size, false) {
                        let (input_batch, target_batch) = batch?;
                        let loss = self
                            .entropy
                            .loss_tensor(model, &input_batch, &target_batch)?;
                        // We dont perform the step here, as its a validation set
                        val_loss_sum += loss.to_scalar::<f32>()?;
                        val_batches += 1;
                    }
                    let val_loss_avg = val_loss_sum / val_batches as f32;
                    println!("Epoch {epoch}/{epochs} Avg Validation Loss: {val_loss_avg}");
                    io::stdout().flush()?
                }
                i += 1;
            }
        }
        Ok(())
    }
}

pub struct CrossEntropy {
    pub device: Device,
    pub train: bool,
    pub ignore_index: Option<i64>,
}

impl Default for CrossEntropy {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            train: false,
            ignore_index: None,
        }
    }
}

impl CrossEntropy {
    pub fn compute(
        &self,
        model: &GPTModel,
        data: TensorDataset,
        batch_size: usize,
        shuffle_batches: bool,
    ) -> eyre::Result<f32> {
        let mut total_loss = 0.;
        let mut count = 0;
        for batch in data.batches(batch_size, shuffle_batches) {
            let (input_batch, target_batch) = batch?;

            let loss = self.loss_scalar(model, &input_batch, &target_batch)?;
            total_loss += loss;
            count += 1;
        }
        let out = total_loss / count as f32;
        Ok(out)
    }

    pub fn loss_tensor(
        &self,
        model: &GPTModel,
        input_batch: &Tensor,
        target_batch: &Tensor,
    ) -> eyre::Result<Tensor> {
        let input_batch = input_batch.to_device(&self.device)?;
        let target_batch = target_batch.to_device(&self.device)?;

        // Forward pass
        let logits = model.forward_t(&input_batch, self.train)?;

        // flatten
        let logits_flat = logits.flatten(0, 1)?;
        let mut targets_flat = target_batch.flatten_all()?;

        if targets_flat.dtype() != candle_core::DType::I64 {
            targets_flat = targets_flat.to_dtype(candle_core::DType::I64)?;
        }

        // Optionally filter out ignored indices (e.g., -100)
        let (logits_flat, targets_flat) = self.filter_indices(logits_flat, targets_flat)?;

        // Forward pass
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
        Ok(loss)
    }

    pub fn loss_scalar(
        &self,
        model: &GPTModel,
        input_batch: &Tensor,
        target_batch: &Tensor,
    ) -> eyre::Result<f32> {
        let loss = self.loss_tensor(model, input_batch, target_batch)?;
        let loss = loss.to_scalar::<f32>()?;
        Ok(loss)
    }

    fn filter_indices(
        &self,
        logits_flat: Tensor,
        targets_flat: Tensor,
    ) -> eyre::Result<(Tensor, Tensor)> {
        let Some(ignore_val) = self.ignore_index else {
            return Ok((logits_flat, targets_flat));
        };

        // get indices to keep
        let keep = targets_flat
            .to_vec1::<i64>()? // has to be i64 to include ignore_index of -100
            .iter()
            .enumerate()
            .filter(|(_, v)| v != &&ignore_val)
            .map(|(ix, _)| ix as u32)
            .collect::<Vec<_>>();
        let keep = Tensor::new(&keep[..], &self.device)?;

        let logits_flat = logits_flat.index_select(&keep, 0)?;
        let targets_flat = targets_flat.index_select(&keep, 0)?;

        Ok((logits_flat, targets_flat))
    }
}
