package deepDriver.dl.aml.lstm;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;


import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.ann.imp.LogicsticsActivationFunction;
import deepDriver.dl.aml.lstm.imp.Block;
import deepDriver.dl.aml.lstm.imp.TanhAf;

public class BPTT implements IBPTT {
	
	LSTMConfigurator cfg;	
	int firstLstmPos = 1;
	
	//New CRF Layer
	CRFLayer crfLayer;
	boolean isCRF;
	
	Context cxt;
	Context deltaCxt;
	public BPTT(LSTMConfigurator cfg) {
		super();
		this.cfg = cfg;
		this.m = cfg.m;
		this.learningRate = cfg.learningRate;
		this.mibatchSize = cfg.miniBatchSize;
		this.dropOut = cfg.dropOut;
		f = new LogicsticsActivationFunction();
		g = h = new TanhAf();  
		if (!(cfg.layers[firstLstmPos] instanceof LSTMLayer)) {
			firstLstmPos = firstLstmPos + 1;			
		}
		_preCxtSc = new double[cfg.layers[firstLstmPos].getRNNNeuroVos().length];
		_preCxtAa = new double[cfg.layers[firstLstmPos].getRNNNeuroVos().length];
		
		this.enableUseCellAa = cfg.enableUseCellAa;
		this.crfLayer = cfg.crfLayer;
		this.isCRF = cfg.crf;
	}
	
	LstmAttention attention;
	double [][] attentionDhj;
	
	int t;
	double [] feature;
	double [] target;
	int layerPos;
	
	boolean isTesting = false; 	
	public LstmAttention getAttention() {
		return attention;
	}

	public void setAttention(LstmAttention attention) {
		this.attention = attention;
	} 

	public double[][] getAttentionDhj() {
		return attentionDhj;
	}

	public void setAttentionDhj(double[][] attentionDhj) {
		this.attentionDhj = attentionDhj;
	}

	public double [][] fTT(double [][] sample, boolean test) {
		isTesting = test;
		tLength = sample.length;
		return fTT(sample);
	}	
	
	public Context getDeltaCxt() {
		return deltaCxt;
	}

	public void setDeltaCxt(Context deltaCxt) {
		this.deltaCxt = deltaCxt;
	}

	double [][] sample;
	
	protected double [][] fTT(double [][] sample) {
		this.sample = sample;
		for (int i = 0; i < sample.length; i++) {
			t = i;
			feature = sample[t];			
			for (int j = 0; j < cfg.layers.length; j++) {
				layerPos = j;
				cfg.layers[j].fTT(this);
			}
		}
		IRNNNeuroVo [] nvs = cfg.layers[cfg.layers.length - 1].getRNNNeuroVos();
		double [][] results = new double[t + 1][nvs.length];
		for (int i = 0; i < results.length; i++) {
			results[i] = new double[nvs.length];
			for (int j = 0; j < results[i].length; j++) {
				results[i][j] = nvs[j].getNvTT()[i].aA;
			}
		}
		return results;
	}
	
	/**
	 * New fTT method for CRF Layer
	 */
	protected double fTTCRF(double [][] sample, double [][] target) {
		double[][] zZ = fTTZz(sample);
		crfLayer.initDelta(sample.length);
		double cost = crfLayer.forward(zZ, target);
		return cost;
	}
	
	/**
	 * New fTT method during inference stage, find optimal sequence
	 */
	protected int[] fTTInfer(double[][] sample) {
		double[][] zZ = fTTZz(sample);
		// double[] z = retreiveTopLayerZzs();
		// Update the delta of CRF Layer
		crfLayer.initDelta(sample.length);
		int[] path = crfLayer.predict(zZ);
		return path;
	}
	
	protected double [][] fTTZz(double [][] sample) {
		this.sample = sample;
		for (int i = 0; i < sample.length; i++) {
			t = i;
			feature = sample[t];			
			for (int j = 0; j < cfg.layers.length; j++) {
				layerPos = j;
				cfg.layers[j].fTT(this);
			}
		}
		IRNNNeuroVo [] nvs = cfg.layers[cfg.layers.length - 1].getRNNNeuroVos();
		double [][] results = new double[t + 1][nvs.length];
		double [][] zZ = new double[t + 1][nvs.length];
		for (int i = 0; i < results.length; i++) {
			results[i] = new double[nvs.length];
			for (int j = 0; j < results[i].length; j++) {
				results[i][j] = nvs[j].getNvTT()[i].aA;
				zZ[i][j] = nvs[j].getNvTT()[i].zZ;
			}
		}
		return zZ;
	}
	
	/**
	 * fTT for gradient check 
	 */
	
	protected double fTTCheck(double[][] zZ, double [][] target, CRFLayer crfLayer) {
		crfLayer.initDelta(sample.length);
		double cost = crfLayer.forward(zZ, target);
		return cost;
	}
	
	double error;
	public double caculateError(double [] target, 
			RNNNeuroVo [] vos, int t) {		
		double stdError = 0;
		if (LSTMConfigurator.SOFT_MAX == cfg.costFunction) {
			for (int i = 0; i < target.length; i++) {
				if (target[i] == 1) {
					SimpleNeuroVo vo = vos[i].getNvTT()[t];	
					stdError = - Math.log(vo.aA);
				}
			}
			return stdError;
		} else {			
			for (int i = 0; i < vos.length; i++) {
				SimpleNeuroVo vo = vos[i].getNvTT()[t];				
				double resisal = target[i] - vo.aA;
				stdError = stdError + resisal * resisal;				
			}		
			return stdError/2.0;
		}
		
	}
	
	boolean isSimpleCxt = false;
	public Context getHLContext() {
		if (isSimpleCxt) {
			return getSimpleHLContext();
		} else {
			Context cxt = new Context();
			int lstmLayerNum = getLstmLayerLength();			
			ContextLayer [] cxtLayers = new ContextLayer[lstmLayerNum];
			for (int i = 0; i < cxtLayers.length; i++) {
				cxtLayers[i] = getHLContext(firstLstmPos + i); 
			}
			cxt.setContextLayers(cxtLayers);
			return cxt;
		}
	}
	
	public int getLstmLayerLength() {
		int lstmLayerNum = cfg.getLayers().length - (firstLstmPos) - 1;
		if (!cfg.isRequireLastRNNLayer()) {
			lstmLayerNum = cfg.getLayers().length - (firstLstmPos);
		}
		return lstmLayerNum;
	}	
	
	public int getFirstLstmPos() {
		return firstLstmPos;
	}

	public void setFirstLstmPos(int firstLstmPos) {
		this.firstLstmPos = firstLstmPos;
	}

	public ContextLayer getHLContext(int pos) {
		IRNNLayer lstmL = cfg.layers[pos];
		IRNNNeuroVo [] cells = lstmL.getRNNNeuroVos();
		double [] rsc = new double[cells.length];
		double [] aas = new double[cells.length];
		boolean isLstm = false;
		if (lstmL instanceof LSTMLayer) {
			isLstm = true;
		}
		for (int i = 0; i < rsc.length; i++) {
			if (isLstm) {
				ICell cell = (ICell)(cells[i]);
				rsc[i] = cell.getSc()[tLength - 1];
			}			
			aas[i] = cells[i].getNvTT()[tLength - 1].aA;
		}
		return new ContextLayer(rsc, aas);
	}
	
	public Context getSimpleHLContext() {
		IRNNLayer lstmL = cfg.layers[firstLstmPos];
		IRNNNeuroVo [] cells = lstmL.getRNNNeuroVos();
		double [] rsc = new double[cells.length];
		double [] aas = new double[cells.length];
		boolean isLstm = false;
		if (lstmL instanceof LSTMLayer) {
			isLstm = true;
		}
		for (int i = 0; i < rsc.length; i++) {
			if (isLstm) {
				ICell cell = (ICell)(cells[i]);
				rsc[i] = cell.getSc()[tLength - 1];
			}			
			aas[i] = cells[i].getNvTT()[tLength - 1].aA;
		}
		return new Context(rsc, aas);
	}
	
	int batchCnt = 0;
	int mibatchSize = 1;
	int tLength;
	
	int ngram = 0;
	
	/**
	 * Original Method runEpich
	 * */
	public double runEpich(double [][] sample, 
			double [][] targets) {
		if (isCRF) {
			double cost = runEpichCRF(sample, targets);
			return cost;
		}
		
		tLength = sample.length;
		fTT(sample, false);
		
		bptt(targets);
		
		if (!cfg.isMeasureOnly()) {
			updateWws();
		}		
		return error;
	}

	public double bptt(double [][] targets) {
		error = 0;
		for (int i = (targets.length - 1); i >= ngram ; i--) {
			t = i;			
			target = targets[t];
			if (cfg.isRequireLastRNNLayer()) {
				error = error + caculateError(target, cfg.layers[cfg.layers.length -1].getRNNNeuroVos(),
					t);
			}
			
			for (int j = (cfg.layers.length -1); j >= 0 ; j--) {
				layerPos = j;
				cfg.layers[j].bpTT(this);
			}
		}
		return error;
	}	
	
	/**
	 * New Method for LSTM-CRF Model
	 * */
	
	public double runEpichCRF(double [][] sample, 
			double [][] targets) {		
		tLength = sample.length;
		// forward Pass: LSTM -> CRF
		double cost = fTTCRF(sample, targets);
		
		// bptt(targets);
		bpttCRF(targets);  // pass dZ back

		// Gradient Check, after bptt dA and dZ
		boolean check = false;
		if (check) {
			boolean pass = gradCheck(sample, targets);
			System.out.println("Gradient Check Passed: " + pass);
		}		
		
		if (!cfg.isMeasureOnly()) {
			updateWws();
		}
		return cost;
	}
	
	/**
	 * New Method for Gradient Check
	 * */
	
	private boolean gradCheck(double [][] sample, double [][] target) {
		boolean pass = true;
		double eps = 0.0001;
		double threshold = 0.001;
		double[][] A = copyArray(crfLayer.getA());
		double[][] dA = copyArray(crfLayer.getdA());	
		
		int K = A.length;
		int T = sample.length;
		double[][] dA_est = new double[K][K]; // dA estimation
		CRFLayer crfLayer = new CRFLayer(K);
		
		// s_Plus and s_Minus use same zZ
		double[][] zZ = fTTZz(sample);
		double[][] dZ = crfLayer.getdZ();
		double[][] dZ_est = new double[T][K];
		
		// Gradient Check AA
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < K; j++) {
				// cur A[i, j] parameters
				double[][] A_plus = copyArray(A);
				double[][] A_minus = copyArray(A);
				A_plus[i][j] += eps;
				A_minus[i][j] -= eps;
				
				// Calc Cost
				crfLayer.setA(A_plus);
				double s_plus = fTTCheck(zZ, target, crfLayer);
				crfLayer.setA(A_minus);
				double s_minus = fTTCheck(zZ, target, crfLayer);
				dA_est[i][j] = (s_plus - s_minus)/(2 * eps);
				//Absolute value because double is not accurate
				double error = Math.abs((dA[i][j] - dA_est[i][j])/dA_est[i][j]);
				if (error < threshold){
					//System.out.println("Check passed " + i + "," + j + ":" + error);
				} else {
					System.out.println("Check failed " + i + "," + j + ":" + error);
					pass = false;
					return pass;
				}
			}
		}
		
		// Gradient Check zZ[t, i]
		for (int t = 0; t < T; t++) {
			for (int i = 0; i < K ; i++) {
				// Refresh for every parameters
				double[][] zZ_plus = copyArray(zZ);
				double[][] zZ_minus = copyArray(zZ);
				zZ_plus[t][i] += eps;
				zZ_minus[t][i] -= eps;
				double s_plus = fTTCheck(zZ_plus, target, crfLayer); // original crf layer
				double s_minus = fTTCheck(zZ_minus, target, crfLayer);
				dZ_est[t][i] = (s_plus - s_minus)/(2 * eps);
				double error = Math.abs((dZ[t][i] - dZ_est[t][i])/dZ_est[t][i]);
				if (error < threshold) {
					// System.out.println("Check passed " + i + "," + t + " Error:" + error);
				} else {	
					System.out.println("Check failed " + i + "," + t + " Error:" + error);
					pass = false;
					return pass;
				}
			}
		}
		return pass;
	}
	
	private double[][] copyArray(double[][] a) {
		double[][] b = new double[a.length][a[0].length];
		for (int i = 0; i < a.length; i++)
			for (int j = 0; j < a[0].length; j++)
				b[i][j] = a[i][j];
		return b;
	}
	
	private double[] copyArray(double[] a) {
		double[] b = new double[a.length];
		for (int i = 0; i < a.length; i++)
			b[i] = a[i];
		return b;
	}	
	
	public double bpttCRF(double [][] targets) {
		error = 0;
		// backward Pass: LSTM -> CRF, crf-dZ
		double[][] dZz = crfLayer.backward(targets);
		// Objective: Cross-Entropy + crf-dZ
		for (int i = (targets.length - 1); i >= ngram ; i--) {
			t = i; // get dZ of time step t
			target = targets[t];
			setDzZs4TopLayer(dZz[t]);
			
			for (int j = (cfg.layers.length -1); j >= 0 ; j--) {
				layerPos = j;
				cfg.layers[j].bpTT(this);
			}
		}
		return error;
	}
	
	static Random random = new Random(System.currentTimeMillis());
	
	public boolean isHiddenLayer(int lp) {
		if (lp > 0 && lp < cfg.layers.length - 1) {
			return true;
		}
		return false;
	}
	
	public void fTT4PartialRNNLayer(RNNNeuroVo [] vos, RNNNeuroVo [] previousVos, int offset, int length) {
		for (int i = offset; i < offset + length; i++) {
			RNNNeuroVo vo = vos[i];
			double zZ = 0;			
			int binaryPos = getBinaryPos(layerPos - 1, previousVos);
			if (binaryPos >= 0) {
				zZ = zZ + vo.wWs[binaryPos];
			} else {
				for (int j = 0; j < previousVos.length; j++) {
					zZ = zZ + previousVos[j].getNvTT()[t].aA * vo.wWs[j];
				}
			}			
			zZ = zZ + vo.wWs[previousVos.length];
			if (isHiddenLayer(layerPos)) {
				zZ = zZ + fTTRecurrentAa(vo, vos);
			}
			vo.neuroVos[t].zZ = zZ;
			vo.neuroVos[t].aA = f.activate(zZ); 
		}
	}
	
	public void fTT4PartialRNNLayer(RNNNeuroVo [] vos, RNNNeuroVo [] previousVos) {
		fTT4PartialRNNLayer(vos, previousVos, 0, vos.length);
	}

	@Override
	public void fTT4RNNLayer(RNNLayer layer) {
		if (layerPos == 0) {			
			if (cfg.isUseThinData()) {
				return;//use thin data from sample directly.
			}			
			RNNNeuroVo [] vos = layer.getRNNNeuroVos();
			for (int i = 0; i < vos.length; i++) {				
				vos[i].neuroVos[t].aA = feature[i];
			}
			return;
		}
		RNNNeuroVo [] vos = layer.getRNNNeuroVos();
		RNNNeuroVo [] previousVos = cfg.layers[layerPos - 1].getRNNNeuroVos();
		
		fTT4PartialRNNLayer(vos, previousVos);		
		
		if (layerPos == cfg.layers.length - 1) {
			activateLastLayer(layer.getRNNNeuroVos());
		}
	}
	
	public void activateLastLayer(RNNNeuroVo [] vos) {
		if (LSTMConfigurator.SOFT_MAX == cfg.costFunction) {
			double [] yt = new double[vos.length];
			double sum = 0;
			for (int i = 0; i < vos.length; i++) {
				SimpleNeuroVo vo = vos[i].getNvTT()[t];
				yt[i] = Math.exp(vo.zZ);
				sum = sum + yt[i];
			}		
			for (int i = 0; i < vos.length; i++) {
				SimpleNeuroVo vo = vos[i].getNvTT()[t];
				vo.aA = yt[i]/sum;
			}
		} 
	}
	
	IActivationFunction f;
	IActivationFunction g;
	IActivationFunction h;
	
	double [] _preCxtSc = null;	
	double [] _preCxtAa = null;	
	
	public double[] cxtDeltaZz(int layerPos) {
		if (deltaCxt == null) {
			return _preCxtAa;
		}
		if (deltaCxt.getContextLayers() != null) {
			return deltaCxt.getContextLayers()[layerPos - firstLstmPos].getPreCxtAa();
		} else {
			return deltaCxt.getPreCxtAa();
		}//
	}
	
	public double[] cxtDeltaSc(int layerPos) {
		if (deltaCxt == null) {
			return _preCxtSc;
		}
		if (deltaCxt.getContextLayers() != null) {
			return deltaCxt.getContextLayers()[layerPos - firstLstmPos].getPreCxtSc();
		} else {
			return deltaCxt.getPreCxtSc();
		}//
	}
	
	public double[] preCxtSc(int layerPos) {
		if (cxt == null) {
			return _preCxtSc;
		}
		if (cxt.getContextLayers() != null) {
			return cxt.getContextLayers()[layerPos - firstLstmPos].getPreCxtSc();
		} else {
			return cxt.getPreCxtSc();
		}//
	}
	
	public double[] preCxtAa(int layerPos) {
		if (cxt == null) {
			return _preCxtAa;
		}
		if (cxt.getContextLayers() != null) {
			return cxt.getContextLayers()[layerPos - firstLstmPos].getPreCxtAa();
		} else {
			return cxt.getPreCxtAa();
		}//
	}


	public Context getPreCxts() {
//		Context ctx = new Context(preCxtSc, preCxtAa);
		return this.cxt;
	}

	public void setPreCxts(Context cxt) {
		this.cxt = cxt;
//		this.preCxtSc = cxts.getPreCxtSc();
//		this.preCxtAa = cxts.getPreCxtAa();
	}
	
	boolean enableUseCellAa = true;
	

	public boolean isEnableUseCellAa() {
		return enableUseCellAa;
	}

	public void setEnableUseCellAa(boolean enableUseCellAa) {
		this.enableUseCellAa = enableUseCellAa;
	}
	
	public int getBinaryPos(int previousLayerPos, RNNNeuroVo [] previousVos) {
		int binaryPos = -1;
		if (cfg.binaryLearning) {
			if (previousLayerPos == 0) {
//				speedUpLearning = true;
				if (cfg.isUseThinData()) {
					binaryPos = (int) sample[t][0];
				} else {
					for (int j = 0; j < previousVos.length; j++) {
						SimpleNeuroVo sno = previousVos[j].neuroVos[t];
						if (sno.aA == 1) {
							binaryPos = j;
							break;
						}
					}
				}				
			}
		}
		return binaryPos;
	}
	
	public void fTT4PartialLstmLayer(Block [] blocks, int offset, int length, RNNNeuroVo [] previousVos, LSTMLayer layer, int binaryPos, boolean speedUpLearning) {
		for (int i = offset; i < offset + length; i++) {
			Block block = blocks[i];	
			int abs = 0;
			if (useAbsoluteSc) {
				abs = i;
			}
			
			fTTiRNNNeuroVo(block.getInputGate(), previousVos, block, 1, binaryPos, speedUpLearning, layer, abs);
			fTTiRNNNeuroVo(block.getForgetGate(), previousVos, block, 1, binaryPos, speedUpLearning, layer, abs);
			ICell[] cells = block.getCells();
			
			for (int j = 0; j < cells.length; j++) {				
				ICell cell = cells[j];
				SimpleNeuroVo snv = cell.getNvTT()[t];
				/**
				 * <apply drop out>
				 * ***/
				if (dropOut > 0) {
					if (!isTesting) {
						if (random.nextDouble() > dropOut) {
							snv.dropOut = false;
						} else {
							snv.dropOut = true;
							snv.zZ = 0;
							snv.aA = 0;
							cell.getCZz()[t] = 0;
							cell.getSc()[t] = 0;	
							continue;//no need caculation.
						}						
					} 					
				}
				/**
				 * </apply drop out>
				 * ***/
				double zZ = 0;
				/****/
				if (speedUpLearning) {
					zZ = zZ + cell.getwWs()[binaryPos];
				} else {
					for (int k = 0; k < previousVos.length; k++) {
						zZ = zZ + previousVos[k].neuroVos[t].aA * cell.getwWs()[k];
					}
				}	
				/****/
				if (cfg.isUseBias()) {
					zZ = zZ + cell.getwWs()[cell.getwWs().length - 1];
				}				
				/****
				 * <add the activation of last moment>
				 * **/
				zZ = zZ + fTTRecurrentAa(cell, layer.getCells());				
				/****
				 * </add the activation of last moment>
				 * **/
				cell.getCZz()[t] = zZ;
				double sc = block.getInputGate().getNvTT()[t].aA * g.activate(zZ);				
				if (t > 0) {
					sc = sc + block.getForgetGate().getNvTT()[t].aA * 
							cell.getSc()[t - 1];
				} else {
					/**
					 * <0 phase sc>
					 * */					
					sc = sc + block.getForgetGate().getNvTT()[t].aA * 
							preCxtSc(layerPos)[abs + j];
					/**
					 *  </0 phase sc>
					 * */
				}
				if (attention != null) {
					sc = sc + attention.fttAttentionSc(layer, (RNNNeuroVo) cell, t);
				}
				cell.getSc()[t] = sc;	
				/**
				 * <apply drop out>
				 * ***/
				if (dropOut > 0) {
					if (isTesting) {
						cell.getSc()[t] = cell.getSc()[t] * (1.0 - dropOut);
					}
				}
				/**
				 * </apply drop out>
				 * ***/
			}			
			fTTiRNNNeuroVo(block.getOutPutGate(), previousVos, block, 0, binaryPos, speedUpLearning, layer, abs);
			for (int j = 0; j < cells.length; j++) {
				ICell cell = cells[j];
				SimpleNeuroVo snv = cell.getNvTT()[t];
				snv.aA = block.getOutPutGate().getNvTT()[t].aA 
						* h.activate(cell.getSc()[t]);
				/**
				 * <apply drop out>
				 * ***/
				if (dropOut > 0) {
					if (!isTesting) {
						if (snv.dropOut) {//it is checked before.
							snv.zZ = 0;
							snv.aA = 0;
							cell.getCZz()[t] = 0;
							cell.getSc()[t] = 0;	
						}						
					} else {
						snv.aA = snv.aA * (1.0 - dropOut);
					}
				}
				/**
				 * </apply drop out>
				 * ***/
			}
		}
	}
	
	public void fTT4PartialLstmLayer(Block [] blocks, RNNNeuroVo [] previousVos, LSTMLayer layer, int binaryPos, boolean speedUpLearning) {
		fTT4PartialLstmLayer(blocks, 0, blocks.length, previousVos, layer, binaryPos, speedUpLearning);
	}

	@Override
	public void fTT4RNNLayer(LSTMLayer layer) {
		if (attention != null) {
			attention.fTT4RNNLayerAttention(layer, t);
		}
		if (layerPos == 0) {
			return;
		}
		int previousLayerPos = layerPos - 1;
		RNNNeuroVo [] previousVos = cfg.layers[previousLayerPos].getRNNNeuroVos();
		/****/
		int binaryPos = 0;
		boolean speedUpLearning = false;
		binaryPos = getBinaryPos(previousLayerPos, previousVos);
		if (binaryPos >= 0) {
			speedUpLearning = true;
		}
		/****/
		
		Block [] blocks = layer.getBlocks();
		fTT4PartialLstmLayer(blocks, previousVos, layer, binaryPos, speedUpLearning);
	}
	
	public double fTTRecurrentAa(IRNNNeuroVo vo, IRNNNeuroVo [] currentVos) {
		double zZ = 0;
		if (enableUseCellAa) {
			if (t > 0) {
				for (int k = 0; k < currentVos.length; k++) {
					IRNNNeuroVo lastTCell = currentVos[k];
					zZ = zZ + vo.getLwWs()[k] * 
							lastTCell.getNvTT()[t - 1].aA;
				}
			} else {
				for (int k = 0; k < currentVos.length; k++) {
					zZ = zZ + vo.getLwWs()[k] * 
							preCxtAa(layerPos)[k];
				}
			}			
		}
		return zZ;
	}
	
	public double bpTTRecurrentAa(int pos, IRNNNeuroVo [] currentVos) {
		double s = 0;
		if (enableUseCellAa && t < tLength -1) {
			for (int k = 0; k < currentVos.length; k++) {
				IRNNNeuroVo lastTCell = currentVos[k];
				s = s + lastTCell.getNvTT()[t + 1].deltaZz
						* lastTCell.getLwWs()[pos];
			}					
		}
		return s;
	}
	
	public double bpTTUseCellAa(int pos, ICell[] cells) {
		double s = 0;
		if (enableUseCellAa && t < tLength -1) {
			for (int k = 0; k < cells.length; k++) {
				ICell lastTCell = cells[k];
				s = s + lastTCell.getDeltaC()[t + 1]
						* lastTCell.getLwWs()[pos];
			}					
		}
		return s;
	}
	
	boolean useCAa4Gate = true;
	public double bpTTUseBlocks(int pos, IBlock [] blocks) {
		double s = 0;
		if (useCAa4Gate && t < tLength -1) {
			for (int k = 0; k < blocks.length; k++) {
				IBlock lastTBlock = blocks[k]; 
				SimpleNeuroVo fVo_t = getIRNNNeuroVo(lastTBlock.getForgetGate(), t + 1); 	
				SimpleNeuroVo iVo_t = getIRNNNeuroVo(lastTBlock.getInputGate(), t + 1);
				SimpleNeuroVo oVo_t = getIRNNNeuroVo(lastTBlock.getOutPutGate(), t + 1); 
				s = s + fVo_t.deltaZz
						* lastTBlock.getForgetGate().getLwWs()[pos]
					+ iVo_t.deltaZz
					* lastTBlock.getInputGate().getLwWs()[pos]
					+ oVo_t.deltaZz
					* lastTBlock.getOutPutGate().getLwWs()[pos];
			}					
		}
		return s;
	}
	
	public void fTTiRNNNeuroVo(IRNNNeuroVo vo, RNNNeuroVo [] previousVos, Block block, 
			int scOffset, int binaryPos, boolean speedUpLearning, LSTMLayer layer, int abs) {
		double zZ = 0;
		if (speedUpLearning) {
			zZ = zZ + vo.getwWs()[binaryPos];
		} else {
			for (int j = 0; j < previousVos.length; j++) {
				zZ = zZ + previousVos[j].getNvTT()[t].aA * vo.getwWs()[j];
			}
		}
		
		ICell[] cells = block.getCells();
		for (int j = 0; j < cells.length; j++) {
			if (t >= scOffset) {
				zZ = zZ + cells[j].getSc()[t - scOffset]
						* vo.getRwWs()[j];
			} else {
				/**
				 * <0 phase sc>
				 * */
				zZ = zZ + preCxtSc(layerPos)[abs + j]
						* vo.getRwWs()[j];
				/**
				 * </0 phase sc>
				 * */
			}
		}
		
		if (cfg.isUseBias()) {
			zZ = zZ + vo.getwWs()[vo.getwWs().length - 1];
		}		
		/****
		 * <add the activation of last moment>
		 * **/
//		zZ = zZ + fTTUseCellAa(vo, cells);
		if (useCAa4Gate) {
			zZ = zZ + fTTRecurrentAa(vo, layer.getCells());	
		}
//		
		/****
		 * </add the activation of last moment>
		 * **/
		vo.getNvTT()[t].zZ = zZ;
		vo.getNvTT()[t].aA = f.activate(zZ);
	}
	
	public void dzZ4TopLayer() {
		if (dzZs4TopLayer != null) {//assuming it is a RNN layer, otherwise this is not correct.
			IRNNLayer layer = cfg.getLayers()[cfg.layers.length - 1];
			RNNNeuroVo [] vos = layer.getRNNNeuroVos();
 			for (int i = 0; i < dzZs4TopLayer.length; i++) {
				RNNNeuroVo vo = vos[i];
				vo.getNvTT()[tLength - 1].deltaZz += dzZs4TopLayer[i];  // cross entropy + transition
			}
		}
	}

	double learningRate = 0.01;
	double m = 0.8;
	double dropOut = 0;
	public void caculateWithCostFunction(RNNNeuroVo [] vos) {		
		if (LSTMConfigurator.SOFT_MAX == cfg.costFunction) {
			double [] yt = new double[vos.length];
			double sum = 0;
			int k = 0;
			for (int i = 0; i < vos.length; i++) {
				SimpleNeuroVo vo = vos[i].getNvTT()[t];
				if (target[i] == 1) {
					k = i;					
				}
				yt[i] = Math.exp(vo.zZ);
				sum = sum + yt[i];
			}		
//			yt = yt / sum;
			for (int i = 0; i < vos.length; i++) {
				SimpleNeuroVo vo = vos[i].getNvTT()[t];
				vo.aA = yt[i]/sum;
				if (k == i) {
					vo.deltaZz = vo.aA - (1.0) ;//* f.deActivate(vo.zZ);	
				} else {
					vo.deltaZz = vo.aA ;//* f.deActivate(vo.zZ);	
				}
			}
		} else {//assume it is least square always.
			for (int i = 0; i < vos.length; i++) {				
				SimpleNeuroVo vo = vos[i].getNvTT()[t];
				vo.deltaZz = (vo.aA - target[i]) * f.deActivate(vo.zZ);				
			}
		}
		// Accumulate delta from CRF Layer
		if (dzZs4TopLayer != null) {
			dzZ4TopLayer();
			return;
		}
		
	}
	
	public double [] retreiveTopLayerZzs() {//assuming retreive from rnn layer always.
		IRNNLayer layer = cfg.getLayers()[cfg.layers.length - 1];
		RNNNeuroVo [] vos = layer.getRNNNeuroVos();
		double [] zZs = new double[vos.length];
		for (int i = 0; i < zZs.length; i++) {
			RNNNeuroVo vo = vos[i];
			zZs[i] = vo.getNvTT()[tLength - 1].zZ;
		}
		return zZs;
	}
	
	double [] dzZs4TopLayer;	
	
	public void setDzZs4TopLayer(double[] dzZs4TopLayer) {
		this.dzZs4TopLayer = dzZs4TopLayer;
	} 

	@Override
	public void bpTT4RNNLayer(RNNLayer layer) {
		if (layerPos == cfg.layers.length - 1) {
			RNNNeuroVo [] vos = layer.getRNNNeuroVos();
			caculateWithCostFunction(vos);	
		} else {
			//do we need to bptt for 1st layer?
			if (layerPos == 0) {
				if (cfg.isBp4FirstLayer()) {
					bpttFromNextLayer(layer, false);
				}
				return;
			}
			bpttFromNextLayer(layer, true);
//			RNNNeuroVo [] vos = layer.getRNNNeuroVos();
//			for (int i = 0; i < vos.length; i++) {				
//				SimpleNeuroVo vo = vos[i].getNvTT()[t];
//				RNNNeuroVo [] nextVos = cfg.layers[layerPos + 1].getRNNNeuroVos();
//				double s = 0;
//				for (int j = 0; j < nextVos.length; j++) {
//					SimpleNeuroVo vo1 = nextVos[j].getNvTT()[t];
//					s = s + vo1.deltaZz * nextVos[j].getwWs()[i];
//				}
//				vo.deltaZz = s * f.deActivate(vo.zZ);
//			}
		}		 
	}
	
	public void bpTT4PartialRNNLayerCell(ICell[] allCells, LSTMLayer layer, int offset, int length) {
		for (int i = offset; i < offset + length; i++) {
			ICell cell = allCells[i];
//			double sc = cell.getSc()[t];
			SimpleNeuroVo vo = cell.getNvTT()[t];				
			double s = 0;
//			for (int k = 0; k < nextVos.length; k++) {
//				SimpleNeuroVo vo1 = nextVos[k].getNvTT()[t];
//				s = s + vo1.deltaZz * nextVos[k].getwWs()[i];
//			}
			//<add cell activation>
			s = s + bpTTUseCellAa(i, layer.getCells());/***cells should be from layer***/
			//</add cell activation>
			
			s = s + bpTTUseBlocks(i, layer.getBlocks());
			
			//the deltaZz is re-initialized during next layer.
			vo.deltaZz = vo.deltaZz + s;	
			/*drop out	
			 * **/
			if (dropOut > 0) {
				if (!isTesting) {
					if (vo.dropOut) {//it is checked before.
						vo.deltaZz = 0;		
					}						
				}
			}
			/*drop out	
			 * **/
		}
	}
	
	public void bpTT4PartialRNNLayerCell(ICell[] allCells, LSTMLayer layer) {
		bpTT4PartialRNNLayerCell(allCells, layer, 0, allCells.length);
	}
	
	boolean useAbsoluteSc = false;	
	
	public boolean isUseAbsoluteSc() {
		return useAbsoluteSc;
	}

	public void setUseAbsoluteSc(boolean useAbsoluteSc) {
		this.useAbsoluteSc = useAbsoluteSc;
	}

	public void bpTT4PartialRNNLayerBlocks(Block [] blocks, LSTMLayer layer, int offset, int length) {
		for (int i = offset; i < offset + length; i++) {
			Block block = blocks[i];
			int abs = 0;
			if (useAbsoluteSc) {
				abs = i;
			}			
			ICell[] cells = block.getCells();			
			double outGateDeltaZz = 0;
			IOutputGate outGate = block.getOutPutGate();
			IInputGate inGate = block.getInputGate();
			IForgetGate fGate = block.getForgetGate();
			for (int j = 0; j < cells.length; j++) {
				ICell cell = cells[j];
				double sc = cell.getSc()[t];
				SimpleNeuroVo vo = cell.getNvTT()[t];				
//				double s = 0;
//				for (int k = 0; k < nextVos.length; k++) {
//					SimpleNeuroVo vo1 = nextVos[k].getNvTT()[t];
//					s = s + vo1.deltaZz * nextVos[k].getwWs()[j];
//				}
				//<add cell activation>
//				s = s + bpTTUseCellAa(j, layer.getCells());/***cells should be from layer***/
				//</add cell activation>
//				vo.deltaZz = s;	
				/*drop out	
				 * **/
//				if (dropOut > 0) {
//					if (!isTesting) {
//						if (vo.dropOut) {//it is checked before.
//							vo.deltaZz = 0;		
//						}						
//					}
//				}
				/*drop out	
				 * **/
				outGateDeltaZz = outGateDeltaZz + vo.deltaZz * 
						h.activate(sc) * f.deActivate(
								outGate.getNvTT()[t].zZ);

			}
			getIRNNNeuroVo(outGate, t).deltaZz = outGateDeltaZz;
			for (int j = 0; j < cells.length; j++) {
				ICell cell = cells[j];
				double sc = cell.getSc()[t];
				SimpleNeuroVo vo = getIRNNNeuroVo(cell, t);	 
				double deltaSc = vo.deltaZz * outGate.
				getNvTT()[t].aA * h.deActivate(sc)
				+ outGateDeltaZz * outGate.getRwWs()[j];				
				
//				if (t < lstm.timePeriod - 1) {
				if (t < tLength - 1) {
					SimpleNeuroVo fgVo = getIRNNNeuroVo(fGate, t + 1);	
					SimpleNeuroVo inVo = getIRNNNeuroVo(inGate, t + 1);
					deltaSc = deltaSc + cell.getDeltaSc()[t + 1] * 
							fgVo.aA
							+ fgVo.deltaZz * fGate.getRwWs()[j] + inVo.getDeltaZz() *
							inGate.getRwWs()[j];
				} else {
					/*<delta context>
					 * **/
					if (layerPos == cfg.layers.length - 1) {
						deltaSc = deltaSc + cxtDeltaSc(layerPos)[abs + j];
					}
					/*</delta context>
					 * **/
				}
				if (attention != null && t < tLength - 1) {
					deltaSc = deltaSc + attention.getDeltaSct_1(layer, abs + j);
				}
				cell.getDeltaSc()[t] = deltaSc;
				/*drop out	
				 * **/
				if (dropOut > 0) {
					if (!isTesting) {
						if (vo.dropOut) {//it is checked before.
							cell.getDeltaSc()[t] = 0;		
						}						
					}
				}
				/*drop out	
				 * **/
			}
			
			//caculate the delta error of input gate and forget gate
			SimpleNeuroVo fVo_t = getIRNNNeuroVo(fGate, t);
			double fDeltaZz_t = 0;			
			SimpleNeuroVo iVo_t = getIRNNNeuroVo(inGate, t);
			double iDeltaZz_t = 0;
			for (int j = 0; j < cells.length; j++) {
				ICell cell = cells[j];
				if (t > 0) {
					fDeltaZz_t = fDeltaZz_t + cell.getDeltaSc()[t] * 
							cell.getSc()[t - 1] * f.deActivate(fVo_t.zZ);
				} else {
					/**
					 * 
					 * **/
					fDeltaZz_t = fDeltaZz_t + cell.getDeltaSc()[t] * 
							preCxtSc(layerPos)[abs + j] * f.deActivate(fVo_t.zZ);
					/**
					 * 
					 * **/
				}
				iDeltaZz_t = iDeltaZz_t + cell.getDeltaSc()[t] * 
						g.activate(cell.getCZz()[t]) * f.deActivate(iVo_t.zZ);
				cell.getDeltaC()[t] = cell.getDeltaSc()[t]
						* iVo_t.aA * g.deActivate(cell.getCZz()[t]);
				//
			}
			fVo_t.deltaZz = fDeltaZz_t;
			iVo_t.deltaZz = iDeltaZz_t;
		}
	}
	
	
	public void bpTT4PartialRNNLayerBlocks(Block [] blocks, LSTMLayer layer) {
		bpTT4PartialRNNLayerBlocks(blocks, layer, 0, blocks.length);
	}

	@Override
	public void bpTT4RNNLayer(LSTMLayer layer) {
		if (attention != null) {
			if (t + 1 < tLength) {
				attention.bp4RNNLayerAttention(layer, t + 1);
			}			
		}
		if (layerPos == cfg.layers.length - 1) {			
		}
//		RNNNeuroVo [] nextVos = cfg.layers[layerPos + 1].getRNNNeuroVos();
		ICell[] allCells = layer.getCells();
		if (layerPos != cfg.layers.length - 1) {//if lstm is the last layer, it means no need bp for it.	
			bpttFromNextLayer(layer, false);
		} else {						
			//
			if (attentionDhj != null) {
				for (int j = 0; j < allCells.length; j++) {
					SimpleNeuroVo vo = allCells[j].getNvTT()[t];
					vo.deltaZz = attentionDhj[t][j];
				}
			} else {
				if (t == tLength - 1) {
					for (int j = 0; j < allCells.length; j++) {
						SimpleNeuroVo vo = allCells[j].getNvTT()[t];
						vo.deltaZz = this.cxtDeltaZz(layerPos)[j];
					}
				} else {
					for (int j = 0; j < allCells.length; j++) {
						SimpleNeuroVo vo = allCells[j].getNvTT()[t];
						vo.deltaZz = 0;
					}
				}
			}
			
			//
		}
		
		bpTT4PartialRNNLayerCell(allCells, layer);
		
		Block [] blocks = layer.getBlocks();
		bpTT4PartialRNNLayerBlocks(blocks, layer);
		
	}
	
	public SimpleNeuroVo getIRNNNeuroVo(IRNNNeuroVo nv, int t) {
		return nv.getNvTT()[t];
	}
	
	boolean update = false;
	public void updateWws() {
		if (cfg.isInteractiveUpdate()) {
			mibatchSize = 1;
		} else {
			mibatchSize = cfg.miniBatchSize;
		}		
		batchCnt ++;
		if (batchCnt >= mibatchSize) {
			update = true;
		} else {
			update = false;
		}
		
		/* Update CRF Layer A transition matrix if isCRF is true
		 * */
		if (isCRF) {
			crfLayer.updateWw(cfg.getLearningRate());
		}

		for (int i = 0; i < cfg.layers.length; i++) {
			this.layerPos = i;
			cfg.layers[i].updateWw(this);
		}
		if (batchCnt >= mibatchSize) {
			batchCnt = 0;
		}
	}
	
	public void updateWw4PartialRNNLayer(RNNNeuroVo [] rNNNeuroVos, int offset, int length) {
		IRNNLayer previouLayer = cfg.layers[layerPos - 1];
		for (int i = offset; i < offset + length; i++) {
//			updateWws(rNNNeuroVos[i].getwWs(), rNNNeuroVos[i].getDeltaWWs(), 
//					rNNNeuroVos[i].getNvTT(), previouLayer.getRNNNeuroVos(), null
//					, rNNNeuroVos[i].getxWWs());
			updateWws4Previous(rNNNeuroVos[i].getwWs(), rNNNeuroVos[i].getDeltaWWs(), 
					rNNNeuroVos[i].getNvTT(), previouLayer.getRNNNeuroVos(), null
					, rNNNeuroVos[i].getxWWs());
//			updateWws4Previous(cell.getwWs(), cell.getDeltaWWs(), null, 
//					previouLayer.getRNNNeuroVos(), cell.getDeltaC(), cell.getxWWs());
		}
	}
	
	public void updateWw4PartialRNNLayer(RNNNeuroVo [] rNNNeuroVos) {
		updateWw4PartialRNNLayer(rNNNeuroVos, 0, rNNNeuroVos.length);
	}
	
	@Override
	public void updateWw4RNNLayer(RNNLayer layer) {
		if (layerPos != 0) {
			RNNNeuroVo [] rNNNeuroVos = layer.getRNNNeuroVos();
			updateWw4PartialRNNLayer(rNNNeuroVos);			
		}
	}
	
	int abs = 0;
	public void updateWw4PartialRNNLayer(LSTMLayer layer, IBlock [] blocks, int offset, int length) {
		IRNNLayer previouLayer = cfg.layers[layerPos - 1];
		for (int i = offset; i < offset + length; i++) {
			IBlock block = blocks[i];	
			if (useAbsoluteSc) {
				abs = i;
			}
			ICell [] cells = block.getCells();
			IInputGate igate = block.getInputGate();
			IOutputGate ogate = block.getOutPutGate();
			IForgetGate fgate = block.getForgetGate();
			for (int j = 0; j < cells.length; j++) {
				ICell cell = cells[j];	
				updateWws4Previous(cell.getwWs(), cell.getDeltaWWs(), null, 
						previouLayer.getRNNNeuroVos(), cell.getDeltaC(), cell.getxWWs());
				if (enableUseCellAa) {
					updateWws(cell.getLwWs(), cell.getDeltaLwWs(), null, 
						layer.getCells(), cell.getDeltaC(), 1, cell.getxLwWs());
				}					
			} 
			updateWws4Previous(igate.getwWs(), igate.getDeltaWWs(), igate.getNvTT(), 
					previouLayer.getRNNNeuroVos(), null, igate.getxWWs()); 
			updateWws(igate.getRwWs(), igate.getDeltaRwWs(), igate.getNvTT(), 
					null, null, block.getCells(), 1, igate.getxRwWs()); 
			if (enableUseCellAa) {
				updateWws(igate.getLwWs(), igate.getDeltaLwWs(), igate.getNvTT(), 
						layer.getCells(), null, 1, igate.getxLwWs());
			}
			
			updateWws4Previous(ogate.getwWs(), ogate.getDeltaWWs(), ogate.getNvTT(), 
					previouLayer.getRNNNeuroVos(), null, ogate.getxWWs()); 
			updateWws(ogate.getRwWs(), ogate.getDeltaRwWs(), ogate.getNvTT(), 
					null, null, block.getCells(), 0, ogate.getxRwWs()); 
			if (enableUseCellAa) {
				updateWws(ogate.getLwWs(), ogate.getDeltaLwWs(), ogate.getNvTT(), 
						layer.getCells(), null, 1, ogate.getxLwWs());
			}
			
			updateWws4Previous(fgate.getwWs(), fgate.getDeltaWWs(), fgate.getNvTT(), 
					previouLayer.getRNNNeuroVos(), null, fgate.getxWWs()); 
			updateWws(fgate.getRwWs(), fgate.getDeltaRwWs(), fgate.getNvTT(), 
					null, null, block.getCells(), 1, fgate.getxRwWs());  
			if (enableUseCellAa) {
				updateWws(fgate.getLwWs(), fgate.getDeltaLwWs(), fgate.getNvTT(), 
						layer.getCells(), null, 1, fgate.getxLwWs());
			}
		}
	}
	
	public void updateWw4PartialLstmLayer(LSTMLayer layer, IBlock [] blocks) {
		updateWw4PartialRNNLayer(layer, blocks, 0, blocks.length);
	}
	
	@Override
	public void updateWw4RNNLayer(LSTMLayer layer) {
		if (attention != null) { 
			attention.updateWw(layer);
		}
		if (layerPos != 0) {
			IBlock [] blocks = layer.getBlocks();
			updateWw4PartialLstmLayer(layer, blocks);
		}
	}
	public void updateWws(double [] wWs, double [] deltaWws, SimpleNeuroVo [] nvsTT,
			IRNNNeuroVo [] nvsOfPreviouLayer, double [] deltaZzTT, double [] xWws) {
		updateWws(wWs, deltaWws, nvsTT,
				nvsOfPreviouLayer, deltaZzTT, null, 
				0, 0, xWws);
	}
	
	public void updateWws(double [] wWs, double [] deltaWws, SimpleNeuroVo [] nvsTT,
			IRNNNeuroVo [] nvsOfPreviouLayer, double [] deltaZzTT, int cellAaOffset, double [] xWws) {
		updateWws(wWs, deltaWws, nvsTT,
				nvsOfPreviouLayer, deltaZzTT, null, 
				0, cellAaOffset, xWws);
	}
	
	public void updateWws(double [] wWs, double [] deltaWws, SimpleNeuroVo [] nvsTT,
			IRNNNeuroVo [] nvsOfPreviouLayer, double [] deltaZzTT, ICell [] cellsOfLastMt, 
			int cellScOffset, double [] xWws){
		updateWws(wWs, deltaWws, nvsTT,
				nvsOfPreviouLayer, deltaZzTT, cellsOfLastMt, 
				cellScOffset, 0, xWws);
	}
	
	public void updateWws4Previous(double [] wWs, double [] deltaWws, SimpleNeuroVo [] nvsTT,
			IRNNNeuroVo [] nvsOfPreviouLayer, double [] deltaZzTT, double [] xWws) {
		this.m = cfg.m;
		this.learningRate = cfg.learningRate;
		if (!(layerPos - 1 == 0 && cfg.isUseThinData())) {
			updateWws(wWs, deltaWws, nvsTT, nvsOfPreviouLayer, deltaZzTT, xWws);
			return;
		} 
		Map <Integer, Double> mp = new HashMap<Integer, Double>();
		for (int j2 = ngram; j2 < tLength; j2++) {
			int ki = (int) this.sample[j2][0];
			double deltaZz = 0;
			if (nvsTT != null) {
				deltaZz = nvsTT[j2].deltaZz;
			} else {
				deltaZz = deltaZzTT[j2];
			}
			put2Map(mp, ki, deltaZz);
			
			/***use bias always.
			 * ***/
			if (cfg.isUseBias()) {
				put2Map(mp, wWs.length - 1, deltaZz);
			}			
			/******/
		}
		
		Iterator<Integer> iter = mp.keySet().iterator();
		while (iter.hasNext()) {
			Integer it = (Integer) iter.next();
			updateDelta2Ww(wWs, deltaWws, mp.get(it), xWws, it);
		}
	}
	
	public void put2Map(Map <Integer, Double> mp, int ki, double deltaZz) {
		if (mp.get(ki) != null) {
			double a = mp.get(ki);
			mp.put(ki, a + deltaZz);
		} else {
			mp.put(ki, deltaZz);
		}
	}
	
	public void updateWws(double [] wWs, double [] deltaWws, SimpleNeuroVo [] nvsTT,
			IRNNNeuroVo [] nvsOfPreviouLayer, double [] deltaZzTT, ICell [] cellsOfLastMt, 
			int cellScOffset, int cellAaOffset, double [] xWws) {  	
		this.m = cfg.m;
		this.learningRate = cfg.learningRate;
		for (int j = 0; j < wWs.length; j++) {
			double deltaWw = 0;
//			for (int j2 = 0; j2 < lstm.timePeriod; j2++) {
			for (int j2 = ngram; j2 < tLength; j2++) {
				double deltaZz = 0;
				if (nvsTT != null) {
					deltaZz = nvsTT[j2].deltaZz;
				} else {
					deltaZz = deltaZzTT[j2];
				}
				if (nvsOfPreviouLayer != null) {
					if (j == nvsOfPreviouLayer.length) {
						deltaWw = deltaWw + deltaZz;
					} else {
						if (j2 >= cellAaOffset) {
							deltaWw = deltaWw + deltaZz
								* nvsOfPreviouLayer[j].getNvTT()[j2 - cellAaOffset].aA;
						} else {
							deltaWw = deltaWw + deltaZz
									* preCxtAa(layerPos)[j];
						}						
					}
				} else {
					if (j2 >= cellScOffset) {
						deltaWw = deltaWw + deltaZz
							* cellsOfLastMt[j].getSc()[j2 - cellScOffset];
					} else {
						deltaWw = deltaWw + deltaZz
								* preCxtSc(layerPos)[abs + j];//this is incorrect.
					}					
				}
									
			}
			updateDelta2Ww(wWs, deltaWws, deltaWw, xWws, j);			
		}
	}
	
	public void updateDelta2Ww(double [] wWs, double [] deltaWws, double deltaWw, double [] xWws, int j) {
		if (cfg.isBatchSize4DeltaWw()) {
			deltaWw = - deltaWw/((double)mibatchSize);	
		} else {
			deltaWw = - deltaWw;	
		}					
//		deltaWw =  (1.0 - m) * learningRate * deltaWw + m * deltaWws[j];
		if (cfg.isUseRmsProp()) {
			xWws[j] = 0.9 * xWws[j] + 0.1 * Math.pow(deltaWw, 2);
			deltaWw = deltaWw/(Math.sqrt(xWws[j]) + rmsProp_u);
		}
		if (batchCnt == 1) {
			deltaWw =  learningRate * deltaWw + m * deltaWws[j];
		} else {
			deltaWw =  learningRate * deltaWw + deltaWws[j];
		}
		/***
		 * **/
		double s = deltaWw * deltaWw;
		if (s > gm) {
			deltaWw = deltaWw * gm/s;
//			System.out.println("The deltaWw2 > "+gm+", rescaling to "+deltaWw);
		}
		/***
		 * **/
		if (update) {				
			wWs[j] = wWs[j] + deltaWw;
		}			
		deltaWws[j] = deltaWw;
	}
	
	double rmsProp_u = 0.001;
	
	double gm = 5;
	@Override
	public void updateWw4RNNLayer(ProjectionLayer layer) {
		RNNNeuroVo [] rNNNeuroVos = layer.getRNNNeuroVos();
//		IRNNLayer previouLayer = cfg.layers[layerPos - 1];
		double l = cfg.getLearningRate();
		for (int j2 = ngram; j2 < tLength; j2++) {
			double [] v = layer.w2vList.get(layer.getWd(j2));
			for (int i = 0; i < rNNNeuroVos.length; i++) {
				v[i] = v[i] - l * rNNNeuroVos[i].neuroVos[t].deltaZz;
			}
		}		
	}

	@Override
	public void fTT4RNNLayer(ProjectionLayer layer) {
		int previousLayerPos = layerPos - 1;
		RNNNeuroVo [] previousVos = cfg.layers[previousLayerPos].getRNNNeuroVos();
		/****/
		int binaryPos = 0;
//		boolean speedUpLearning = false;
		binaryPos = getBinaryPos(previousLayerPos, previousVos);
		if (binaryPos < 0) {
			System.out.println("Error, binary pos shouble be larger than 0");
		}
		/****/
		layer.setWd(t, binaryPos);
		double [] v = layer.w2vList.get(binaryPos);
		if (v == null) {
			v = layer.generateV();
			layer.w2vList.put(binaryPos, v);
		}
		RNNNeuroVo [] vos = layer.getRNNNeuroVos(); 
		for (int i = 0; i < vos.length; i++) {
			vos[i].neuroVos[t].aA = v[i];
		}
	}
	
	public void bpttPartialFromNextLayer(RNNNeuroVo [] vos, IRNNLayer layer, boolean useDeActivate, int offset, int length) {
		bpttPartialFromNextLayer(cfg.layers[layerPos + 1],  vos, layer, useDeActivate, offset, length, false);
	}
	
	public void bpttPartialFromNextLayer(IRNNLayer nextLayer,  RNNNeuroVo [] vos, IRNNLayer layer, boolean useDeActivate, boolean addtive) {
		bpttPartialFromNextLayer(nextLayer,  vos, layer, useDeActivate, 0, vos.length, false);
	}
	
	public void bpttPartialFromNextLayer(IRNNLayer nextLayer,  RNNNeuroVo [] vos, IRNNLayer layer, boolean useDeActivate, int offset, int length, boolean addtive) {
		//do we need to reset all the values?
		if (nextLayer instanceof RNNLayer) {
			for (int i = offset; i < offset + length; i++) {
				SimpleNeuroVo vo = vos[i].getNvTT()[t];
				RNNNeuroVo[] nextVos = nextLayer
						.getRNNNeuroVos();
				double s = 0;
				for (int j = 0; j < nextVos.length; j++) {
					SimpleNeuroVo vo1 = nextVos[j].getNvTT()[t];
					s = s + vo1.deltaZz * nextVos[j].getwWs()[i];
				}
				if (layer instanceof RNNLayer && isHiddenLayer(layerPos)) {
					s = s + bpTTRecurrentAa(i, vos);
				}
				if (useDeActivate) {
					vo.deltaZz = s * f.deActivate(vo.zZ);
				} else {
					if (addtive) {
						vo.deltaZz = vo.deltaZz + s;
					} else {
						vo.deltaZz = s;
					}					
				} 
			}
		} else if (nextLayer instanceof LSTMLayer) {
			LSTMLayer nlayer = (LSTMLayer) nextLayer;
			for (int i = offset; i < offset + length; i++) {
				SimpleNeuroVo vo = vos[i].getNvTT()[t];
				Block [] blocks = nlayer.getBlocks();
				double s = 0;
				for (int j = 0; j < blocks.length; j++) {
					Block block = blocks[j];
					ICell [] cells = block.getCells();
					for (int k = 0; k < cells.length; k++) {
						double cz = cells[k].getDeltaC()[t];
						s = s + cz * cells[k].getwWs()[i];
					}
					SimpleNeuroVo vo1 = block.getInputGate().getNvTT()[t];
					s = s + vo1.deltaZz * block.getInputGate().getwWs()[i];
					
					SimpleNeuroVo vo2 = block.getForgetGate().getNvTT()[t];
					s = s + vo2.deltaZz * block.getForgetGate().getwWs()[i];
					
					SimpleNeuroVo vo3 = block.getOutPutGate().getNvTT()[t];
					s = s + vo3.deltaZz * block.getOutPutGate().getwWs()[i];
				}
				if (layer instanceof RNNLayer && isHiddenLayer(layerPos)) {
					s = s + bpTTRecurrentAa(i, vos);
				}
				if (useDeActivate) {
					vo.deltaZz = s * f.deActivate(vo.zZ);
				} else {
//					vo.deltaZz = s;
					if (addtive) {
						vo.deltaZz = vo.deltaZz + s;
					} else {
						vo.deltaZz = s;
					}
				} 
			}
		}
	}
	
	public void bpttPartialFromNextLayer(RNNNeuroVo [] vos, IRNNLayer layer, boolean useDeActivate) {
		bpttPartialFromNextLayer(vos, layer, useDeActivate, 0, vos.length);
	}
	
	public void bpttFromNextLayer(IRNNLayer layer, boolean useDeActivate) {
		RNNNeuroVo [] vos = layer.getRNNNeuroVos();
		bpttPartialFromNextLayer(vos, layer, useDeActivate);
	}

	@Override
	public void bpTT4RNNLayer(ProjectionLayer layer) {
		bpttFromNextLayer(layer, false);		
	}

	

}
