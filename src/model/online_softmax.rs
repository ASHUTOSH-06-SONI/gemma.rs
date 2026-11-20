pub struct OnlineSoftmax {
    pub max: f32,
    pub sum: f32,
}
impl OnlineSoftmax {
    pub fn new() -> Self {
        Self {
            max: f32::NEG_INFINITY,
            sum: 0.0,
        }
    }
    // first calculate the running max and then the update equation (the denominator in softmax function)
    pub fn block_statistics(scores: &[f32]) -> (f32, f32) {
        assert!(!scores.is_empty());
        let mut block_max = f32::NEG_INFINITY;
        for &score in scores {
            block_max = block_max.max(score); // first find the running max
        }
        let mut block_sum = 0.0;
        for &score in scores {
            block_sum += (score-block_max).exp(); // update equation d = d+e^(xi-m) where xi is the score and m is the running max
        }

        (block_max, block_sum)
    }

    // first pass
    // then update the denominator for the softmax function
    pub fn update(&mut self, block_max: f32, block_sum: f32) {
        let new_max = self.max.max(block_max);
        let old_scale = if self.sum == 0.0 {
            0.0
        }else{
            (self.max -new_max).exp()
        };
        let block_scale = (block_max - new_max).exp();
        let new_sum = self.sum * old_scale + block_sum * block_scale;
        self.max = new_max;
        self.sum = new_sum;
    }
    // second pass is just probability calculation for each score
    // e^(xi-m)/d for each i in block
    pub fn probabilities(&self, scores: &[f32]) -> Vec<f32> {
        scores.iter().map(|&score|(score - self.max).exp()/self.sum).collect()
    }
}

