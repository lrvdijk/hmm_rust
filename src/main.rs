extern crate ndarray;
extern crate ndarray_stats;

use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::Array2;
use ndarray_stats::QuantileExt;

struct HMM {
    transitions: Array2<f64>,
    emissions: Array2<f64>,
    initial_state: Array1<f64>
}

impl HMM {
    pub fn new(transitions: Array2<f64>, emissions: Array2<f64>,
               initial_state: Array1<f64>) -> HMM {
        assert_eq!(transitions.shape(), [initial_state.len(), initial_state.len()],
                   "Transitions should be an NxN matrix where N is the number of states (thus the same \
                    size as initial_state).");
        assert_eq!(emissions.shape()[0], initial_state.len(),
                   "First dimension of emissions should be equal to number of states!");

        HMM { transitions, emissions, initial_state }
    }
}

fn viterbi(hmm: &HMM, observations: &[usize]) -> Vec<usize> {
    let mut matrix = Array2::<f64>::zeros((hmm.initial_state.shape()[0], observations.len()));
    let mut pointers = Array2::<usize>::zeros((hmm.initial_state.shape()[0], observations.len()));
    let v_initial = Array1::zeros(hmm.initial_state.len());

    for i in 0..observations.len() {
        let observation_ix = observations[i] - 1;
        assert!(observation_ix < hmm.emissions.shape()[1]);

        let v = if i == 0 {
            v_initial.view()
        } else {
            matrix.column(i - 1)
        };

        let new_v = if i == 0 {
            let prob = hmm.initial_state.view();

            &hmm.emissions.column(observation_ix) + &prob
        } else {
            let prob = &v.slice(s![.., NewAxis]) + &hmm.transitions.t();
            let ptr = &prob.map_axis(Axis(0), |row| row.argmax_skipnan().unwrap());
            pointers.column_mut(i).assign(&ptr);

            &hmm.emissions.column(observation_ix)
                + &prob.map_axis(Axis(0), |row| *row.max_skipnan())
        };

        matrix.column_mut(i).assign(&new_v);
    }

    let uniform = Array1::from_elem(hmm.initial_state.len(), 1./hmm.initial_state.len() as f64).mapv(f64::ln);
    let end_state = (&matrix.column(observations.len() - 1) + &uniform.view()).argmax_skipnan().unwrap();

    let mut state_path = vec![0; observations.len()];
    state_path[observations.len() - 1] = end_state;
    for i in (1..observations.len()).rev() {
        state_path[i-1] = pointers[[state_path[i], i]];
    }

    state_path
}

fn main() {
    let hmm = HMM::new(
        array![[0.95, 0.05], [0.1, 0.9]].mapv(f64::ln),
       array![
            [1./6., 1./6., 1./6., 1./6., 1./6., 1./6.],
            [1./10., 1./10., 1./10., 1./10., 1./10., 1./2.]
        ].mapv(f64::ln),
        array![0.5, 0.5].mapv(f64::ln)
    );

    let observations = [
        3, 1, 5, 1, 1, 6, 2, 4, 6, 4, 4, 6, 6, 4, 4, 2, 4, 5, 3, 1, 1, 3, 2, 1, 6, 3, 1, 1, 6, 4, 1,
        5, 2, 1, 3, 3, 6, 2, 5, 1, 4, 4, 5, 4, 3, 6, 3, 1, 6, 5, 6, 6, 2, 6, 5, 6, 6, 6, 6, 6, 6, 5,
        1, 1, 6, 6, 4, 5, 3, 1, 3, 2, 6, 5, 1, 2, 4
    ];

    let path = viterbi(&hmm, &observations);

    for o in &observations {
        print!("{}", o);
    }
    println!();

    for s in path.iter().map(|v| match v { 0 => "F", 1 => "L", _ => "?" }) {
        print!("{}", s);
    }
}
