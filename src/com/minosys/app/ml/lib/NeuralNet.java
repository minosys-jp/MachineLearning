package com.minosys.app.ml.lib;

import java.io.IOException;
import java.util.Random;
import java.util.stream.IntStream;

import com.minosys.app.ml.lib.SimpleNet.OutputFormat;

/**
 * Neural Network imple,emtations
 * @author minoru
 *
 */
public class NeuralNet {
	/**
	 * learning rate
	 */
	private  final float LRATE;

	/**
	 * layered neurons
	 */
	public SimpleNet[] neurons;

	/**
	 * intermediate output cache (for back propagation)
	 */
	private float[][] z;

	/**
	 * compensation for the back propagation method (weight)
	 */
	private float[][][] wdelta, wderiv;

	/**
	 * compensation for the back propagation method (bias)
	 */
	private float[][] bdelta, bderiv;

	private final Random rand;

	/**
	 * constructor
	 * @param nodes	# of nodes in array format
	 * @throws IOException
	 */
	public NeuralNet(ILossFunction loss, float lrate, int[] nodes) throws IOException {
		this.neurons = new SimpleNet[nodes.length - 1];
		this.z = new float[nodes.length][nodes[nodes.length - 1]];
		this.LRATE = lrate;
		this.rand = new Random();
		IntStream.range(0,  neurons.length).forEach(i->{
			neurons[i] = new SimpleNet(loss, nodes[i], nodes[i + 1]);
			if (i == nodes.length - 2) {
				neurons[i].format = OutputFormat.SOFTMAX;
			}
		});
		wdelta = new float[neurons.length][][];
		bdelta = new float[neurons.length][];
		IntStream.range(0, neurons.length).forEach(k->{
			wdelta[k] = new float[neurons[k].getInn()][neurons[k].getOutn()];
			bdelta[k] = new float[neurons[k].getOutn()];
		});
		bderiv = new float[neurons.length][];
		wderiv = new float[neurons.length][][];
	}

	/**
	 * neural network forward operation
	 * @param in
	 * @return
	 */
	public float[] forward(float[] in) {
		z[0] = in.clone();
		IntStream.range(0, neurons.length).forEachOrdered(k->{
			z[k + 1] = neurons[k].forward(z[k]);
		});
		return z[neurons.length];
	}

	/**
	 * assuming forward() already called and the calculated values are cached
	 * @param ilst
	 * @param index
	 */
	public void backPropagate1(ImageLabelSet ils, int index) {
		// Phase1: calculate the derivatives
		// outer most derivatives
		int m = neurons.length - 1;
		float[] out = ils.label.getContent(index);
		bderiv[m] = neurons[m].calc_deriv_b(out, true);
		wderiv[m] = neurons[m].calc_deriv_w(z[m], bderiv[m]);
		// hidden layer derivatives
		while ( --m >= 0) {
			bderiv[m] = neurons[m].calc_deriv_b(bderiv[m + 1], false);
			wderiv[m] = neurons[m].calc_deriv_w(z[m], bderiv[m]);
		}

		// Phase2: learning process
		IntStream.range(0, neurons.length).forEach(k->{
			IntStream.range(0,  neurons[k].getOutn()).parallel().forEach(j->{
				IntStream.range(0, neurons[k].getInn()).forEach(i->{
					wdelta[k][i][j] += wderiv[k][i][j];
				});
				bdelta[k][j] +=  bderiv[k][j];
			});
		});
	}

	/**
	 * backpropagation for batches
	 * @param ils
	 * @param samples
	 */
	public void backPropagate(ImageLabelSet ils, int[] samples) {
		// initializez deltas
		IntStream.range(0, neurons.length).forEach(k->{
			IntStream.range(0,  neurons[k].getOutn()).parallel().forEach(j->{
				IntStream.range(0, neurons[k].getInn()).forEach(i->{
					wdelta[k][i][j] = 0.0f;
				});
				bdelta[k][j] = 0.0f;
			});
		});

		// phase1: accumlate deltas
		IntStream.range(0, samples.length).forEach(m->{
			// phase1: forward propagation
			forward(ils.image.getContent(samples[m]));
			// phase2: back propagation
			backPropagate1(ils, samples[m]);
		});

		// phase2: learning process
		IntStream.range(0, neurons.length).forEach(k->{
			SimpleNet net = neurons[k];
			IntStream.range(0, net.getOutn()).parallel().forEach(j->{
				IntStream.range(0, net.getInn()).forEach(i->{
					net.w[i][j] -= LRATE * wdelta[k][i][j] / (float)samples.length;
				});
				net.b[j] -= LRATE * bdelta[k][j] / (float)samples.length;
			});
		});
	}

	/**
	 * back propagation calculation for the batch size and the numer of iterations
	 * @param mnist
	 * @param nset
	 * @param batchsize
	 */
	public int[] backPropagate(ImageLabelSet ils, int nset, int batchsize) {
		int[] samples = new int[batchsize];
		IntStream.range(0,  batchsize).forEach(i->{
			samples[i] = rand.nextInt(ils.image.getQuantity());
		});
		IntStream.range(0, nset).forEach(i->{
			backPropagate(ils, samples);
			if (i % 10 == 0) System.out.print(".");
		});
		return samples;
	}
}
