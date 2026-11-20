pub struct AttentionState {
    pub max: f32,
    pub sum: f32,
    pub acc: Vec<f32>,
}

impl AttentionState {
    pub fn new(value_dim: usize) -> Self {
        Self {
            max: f32::NEG_INFINITY,
            sum: 0.0,
            acc: vec![0.0; value_dim],
        }
    }

    pub fn update(
        &mut self,
        block_sum: f32,
        block_max: f32,
        block_acc: &[f32],
    ) {
        let new_max = self.max.max(block_max);

        let alpha = if self.sum == 0.0 {
            0.0
        } else {
            (self.max - new_max).exp()
        };

        let beta = (block_max - new_max).exp();

        for d in 0..self.acc.len() {
            self.acc[d] =
                alpha * self.acc[d] + beta * block_acc[d];
        }

        self.sum = alpha * self.sum + beta * block_sum;
        self.max = new_max;
    }

    pub fn op(&self) -> Vec<f32> {
        self.acc
            .iter()
            .map(|&x| x / self.sum)
            .collect()
    }

    pub fn flash_attn(
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        block_size: usize,
        causal: bool,
    ) -> Result<Vec<Vec<f32>>, String> {
        if q.is_empty() || k.is_empty() || v.is_empty() {
            return Err("Q, K and V must not be empty".to_string());
        }

        if block_size == 0 {
            return Err("Block size must be greater than zero".to_string());
        }

        let key_dim = q[0].len();
        let value_dim = v[0].len();

        if key_dim == 0 || value_dim == 0 {
            return Err("Vector dimensions must not be zero".to_string());
        }

        if k.len() != v.len() {
            return Err("K and V must have the same number of rows".to_string());
        }

        if causal && q.len() != k.len() {
            return Err("Causal attention requires Q and K to have equal sequence lengths".to_string());
        }

        if q.iter().any(|row| row.len() != key_dim)
            || k.iter().any(|row| row.len() != key_dim)
            || v.iter().any(|row| row.len() != value_dim)
        {
            return Err("Inconsistent matrix dimensions".to_string());
        }

        let scale = 1.0 / (key_dim as f32).sqrt();

        let mut output = Vec::with_capacity(q.len());

        for (i, query) in q.iter().enumerate() {
            let mut state = AttentionState::new(value_dim);

            for start in (0..k.len()).step_by(block_size) {
                let end = (start + block_size).min(k.len());

                let mut scores = Vec::with_capacity(end - start);

                for j in start..end {
                    if causal && j > i {
                        continue;
                    }

                    let mut dot_product = 0.0;

                    for d in 0..key_dim {
                        dot_product += query[d] * k[j][d];
                    }

                    scores.push((j, dot_product * scale));
                }

                if scores.is_empty() {
                    continue;
                }

                let block_max = scores
                    .iter()
                    .map(|&(_, score)| score)
                    .fold(f32::NEG_INFINITY, f32::max);

                let mut block_sum = 0.0;
                let mut block_acc = vec![0.0; value_dim];

                for &(j, score) in &scores {
                    let weight = (score - block_max).exp();

                    block_sum += weight;

                    for d in 0..value_dim {
                        block_acc[d] += weight * v[j][d];
                    }
                }

                state.update(block_sum, block_max, &block_acc);
            }

            output.push(state.op());
        }

        Ok(output)
    }
}