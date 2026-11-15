mod model;
use model::online_softmax::OnlineSoftmax;

fn main(){
    let matrix: [[f32; 4]; 4] = [
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 1.0, 1.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
        [10.0, 0.0, -5.0, 2.0],
    ];
    let n = 4;
    let block_size =2 ;
    for i in 0..n{
        let mut state = OnlineSoftmax::new();
        let mut start = 0;
        while start <n{
            let end = (start+block_size).min(n);
            let scores = &matrix[i][start..end];
            let (block_max,block_sum) = OnlineSoftmax::block_statistics(scores);
            state.update(block_max, block_sum);
            start += block_size;
        }
        let probabilities = state.probabilities(&matrix[i]);
        println!("Query {}:{:?}",i,probabilities);
    }
}
