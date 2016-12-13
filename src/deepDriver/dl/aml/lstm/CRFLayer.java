package deepDriver.dl.aml.lstm;

import java.io.Serializable;

public class CRFLayer implements Serializable{
	/**
	 * Serialize CRFLayer object
	 */
	private static final long serialVersionUID = 1L;
	
	int K; // Number of target classes
	double[][] A; //transition matrix, dim: k * k
	
	// CRF Input take output score of LSTM
	int T; // Number of time step
	double[][] delta; // logadd Value, dim: T * k
	double[][] dDelta; // gradient of delta , dim: T * k
	double[][] dA; // gradient of d(Cost)/d(A)
	double[][] dZ; // gradient of d(Cost)/d(out score)
	
	public CRFLayer(int K) {
		this.K = K;
		this.A = new double[K][K];
	}
	
	public void initDelta(int T) {
		this.T = T;
		this.delta = new double[T][this.K];
		this.dDelta = new double[T][this.K];
		this.dA = new double[this.K][this.K];
		this.dZ = new double[T][this.K];
	}
	
	public double forward(double[][] score, double[][] target) {
		double[] y = getMaxPos(target); // class index of each target time step
		double[] yFit = getMaxPos(score);
		int step = score.length;
		fwdRecurrent((step - 1), score); // forward recurrent calc delta
		double logAddSum = logAdd(delta[T-1]);
		double seqSum = sumScore(score, y);
		double cost = logAddSum - seqSum;
		
		// System.out.print("All Log Add Sequence Sum: " + logAddSum + " ");
		// System.out.print("Current Sequence Sum: " + seqSum + " ");
		return cost;
	}
	
	/**
	 * Update the gradient dZ gradient of LSTM backward
	 * */
	
	public double[][] backward(double[][] target) {
		// CalcGradient for A[i,j] and f(theta) score
		double[] y = getMaxPos(target);
		double[][] dA = new double[A.length][A.length];
		double[][] dZ = new double[T][K]; // dZ gradient of LSTM output score
		
		// Gradient of Sum Score Part, -sum(transition + network)
		for (int t = 0; t < T ; t++) { //
			int i = (int) y[t];
			int j;
			dZ[t][i] -= 1.0;
			if (t != (T - 1)) { // if t = T-1, no transition to T yi1
				j = (int) y[t+1];
				dA[i][j] -= 1.0;
			}
		}
		
		//Gradient of the logAdd part, for each time step, update all i,j
		backRecurrent(0); // Back Prop Calc dDelta since t = 0;
		for (int t = 0; t < T; t++) {
			// update dZ all index at time t
			for (int i = 0; i < K ; i++) {
				dZ[t][i] += dDelta[t][i];
			}
			
			// update A[i,j]
			if (t != (T - 1)) {
				for (int k1 = 0; k1 < K; k1++) {
					for (int k2 = 0; k2 < K; k2++) {
						double[] z = softmax(add(delta[t], slice(A, k2)));
						dA[k1][k2] += dDelta[t+1][k2] * z[k1];
					}
				}				
			}
		}
		//Update dZ of layer CRF
		this.dA = dA;
		this.dZ = dZ;
		return dZ;
	}
	
	/**
	 * Find Optimal Path 
	 * */
	int[][] path;  // T * K, save the previous tag id on the path
	public int[] predict(double[][] score) {
		path = new int[T][K];
		int[] optimal_path = new int[T];
		//Viterbi Max calculate max sum score
		double[] delta_t = viterbiMax(T-1, score);
		int pos = getMaxPos(delta_t); // track of position
		optimal_path[T-1] = pos;
		for (int t = (T-1); t > 0; t--) {
			int prev = path[t][pos];
			optimal_path[t-1] = prev;
		}
		return optimal_path;
	}
	
	public void updateWw(double rate) {
		//update transition matrix A
		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A.length; j++) {
				A[i][j] = A[i][j] - rate * dA[i][j];
			}
		}
	}
	
	// Transition score
	private double transScore(double[] y) {
		double s = 0.0;
		for (int i = 0; i < (y.length-1); i++) {
			int yi = (int) y[i];
			int yi1 = (int) y[i+1];
			s += A[yi][yi1];
		}
		return s;
	}
	
	// Tag Network Path Score, Score[it][it_1]
	private double networkScore(double[][] score, double[] y) {
		double s = 0.0;
		int step = score.length;
		for (int t = 0; t < step; t++)
			s += score[t][(int) y[t]];
		return s;
	}
	
	private double sumScore(double[][] score, double[] y){
		double ts = transScore(y);
		double ns = networkScore(score, y);
		double s = ts + ns;
		return s;
	}
	
	private double max(double[] a) {
		double maxVal = Double.MIN_VALUE;
		for (int i = 0; i < a.length; i++)
			if (a[i] > maxVal)
				maxVal = a[i];
		return maxVal;
	}
	
	// substract the max value from array double[] a
	private double[] normalize(double[] a) {
		double[] norm = new double[a.length];
		double maxVal = max(a);
		for (int i = 0; i < a.length; i++)
			norm[i] = a[i] - maxVal;
		return norm;
	}
	
	// Softmax Operation Zi -> exp(Zi)/sum(exp(Zi))
	private double[] softmax(double[] a) {
		double[] aNorm = normalize(a);
		int dim = aNorm.length;
		double[] z = new double[dim];
		double sum = 0.0;
		for (int i = 0; i < dim; i++)
			sum = sum + Math.exp(aNorm[i]);
		for (int i = 0; i < dim; i++)
			z[i] = Math.exp(aNorm[i])/sum;
		return z;
	}
	
	// Log Add Operation, sum(exp(Zi))
	private double logAdd(double[] z) {
		double s = 0.0; // sum
		double log = 0.0; // log(s)
		for (int i = 0; i < z.length; i++)
			s += Math.exp(z[i]);
		log = Math.log(s);
		return log;
	}
	
	private double[] add(double[] a, double[] b){
		double[] z = new double[a.length];
		for (int i = 0; i < a.length; i++) 
			z[i] = a[i] + b[i];
		return z;
	}
	
	private double sum(double[] a){
		double z = 0.0;
		for (int i = 0; i < a.length; i++) 
			z += a[i];
		return z;
	}
	
	private double[] slice(double[][] A, int idx) {
		double[] z = new double[A.length];
		for (int i = 0; i < A.length; i++)
			z[i] = A[i][idx];
		return z;
	}
	
	private int getMaxPos(double[] a) {
		double maxVal = Double.MIN_VALUE;
		int maxPos = -1;
		for (int i = 0; i < a.length; i++) {
			if (a[i] > maxVal) {
				maxVal = a[i];
				maxPos = i;
			}
		}		
		return maxPos;
	}
	
	private double[] getMaxPos(double[][] target) {
		double[] y = new double[target.length];
		for (int i = 0; i < target.length; i++) {
			int pos = getMaxPos(target[i]);
			y[i] = (double) pos;
		}
		return y;
	}
	
	// Viterbi Algorithm Recurrent Calculating LogAddScore
	private double[] fwdRecurrent(int t,double[][] score) {
		if (t == 0) { // delta_t initialized at score[0][k]
			double[] delta_0 = score[0];
			delta[0] = delta_0;
			return delta_0; // transition score 0, tag pathScore score[0];
		} else { // t: 1 to (step-1)
			double[] delta_t = new double[K];
			double[] delta_prev = fwdRecurrent(t-1, score);
			for (int i = 0; i < K; i++) {
				delta_t[i] = score[t][i] + logAdd(add(delta_prev, slice(A, i)));
			}
			delta[t] = delta_t;
			return delta_t;
		}
	}
	
	// Viterbi Algorithm Recurrent Calculating MaxScore
	private double[] viterbiMax(int t,double[][] score) {
		if (t == 0) { // transition score at t = 0, tag pathScore score[0];
			return score[0];
		} else { // t: 1 to (step-1)
			double[] delta_t = new double[K];
			double[] delta_prev = viterbiMax(t-1, score);
			for (int i = 0; i < K; i++) {
				delta_t[i] = score[t][i] + max(add(delta_prev, slice(A, i)));
				// save previous max position
				path[t][i] = getMaxPos(add(delta_prev, slice(A, i)));
			}
			return delta_t;
		}
	}
	
	private double[] backRecurrent(int t) {
		if (t == (T-1)) {
			double[] dDelta_T = softmax(delta[t]);
			dDelta[t] = dDelta_T;
			return dDelta_T;
		} else {
			double[] dDelta_t = new double[K];
			double[] dDelta_next = backRecurrent(t + 1);
			for (int i = 0; i < K; i++) {
				double grad = 0.0;
				for (int j = 0; j < K ; j++) {
					double[] z = softmax(add(delta[t], slice(A, j)));
					grad += (dDelta_next[j] * z[i]);
				}
				dDelta_t[i] = grad;
			}
			dDelta[t] = dDelta_t;
			return dDelta_t;
		}
	}

	public double[][] getA() {
		return A;
	}

	public void setA(double[][] a) {
		A = a;
	}

	public double[][] getdA() {
		return dA;
	}

	public void setdA(double[][] dA) {
		this.dA = dA;
	}

	public double[][] getdZ() {
		return dZ;
	}

	public void setdZ(double[][] dZ) {
		this.dZ = dZ;
	}

}
