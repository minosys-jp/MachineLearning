package com.minosys.app.ml.lib;

import java.util.OptionalDouble;
import java.util.stream.IntStream;

/**
 * simple neural network
 * @author minoru
 *
 */
public class SimpleNet {
	private static final float EPS = (float)1e-2;

	/**
	 * input terminal quantity
	 */
	private final int inn;
	public int getInn() { return inn; }

	/**
	 * output terminal quantity
	 */
	private final int outn;
	public int getOutn() { return outn; }

	/**
	 * weight matrix
	 */
	public float w[][];

	/**
	 * bias
	 */
	public float b[];
	OutputFormat format;

	/**
	 * linear part
	 */
	public float[] z;

	public float[] inz;
	public float[] outz;

	/**
	 * Loss Function
	 */
	public ILossFunction loss;

	/**
	 * output format
	 * @author minoru
	 *
	 */
	enum OutputFormat {
		SIGMOID, SOFTMAX
	};

	/**
	 * Sigmoid function
	 * @param x
	 * @return
	 */
	static float sigmoid(float x) {
		return (float) (1.0f / (1.0f + Math.exp(-x)));
	}

	static float sigmoid_deriv(float x) {
		return x * (1.0f - x);
	}

	/**
	 * calculate the derivative for the biases of the cost function
	 * @param outer
	 * @param w
	 * @param a
	 * @param z
	 * @return
	 */
	public float[] calc_deriv_b(final float[] outer, boolean bMostouter) {

		if (!bMostouter) {
			float[] r = new float[inn];
			// intermediate nodes
			IntStream.range(0, outz.length).parallel().forEach(j->{
				r[j] += IntStream.range(0, outer.length).mapToDouble(k->w[j][k] * outer[k]).sum();
			});
			IntStream.range(0,  inz.length).forEach(j->{
				r[j] *= sigmoid_deriv(inz[j]);
			});
			return r;
		} else {
			float[] r = new float[outn];
			// outer-most nodes
			IntStream.range(0, outz.length).forEach(j->{
				r[j] = loss.derivative(outer[j], outz[j]);
			});
			return r;
		}
	}

	/**
	 * calculate derivative for the weights
	 * @param a
	 * @param delta
	 * @return
	 */
	public float[][] calc_deriv_w(final float[] zz, final float[] delta) {
		float[][] cw = new float[zz.length][delta.length];
		IntStream.range(0,  zz.length).forEach(j->{
			IntStream.range(0,  delta.length).forEach(k->{
				cw[j][k] = zz[j] * delta[k];
			});
		});
		return cw;
	}

	/**
	 * constructor
	 * @param inn
	 * @param outn
	 */
	public SimpleNet(ILossFunction loss, int inn, int outn) {
		this.inn = inn;
		this.outn = outn;
		this.w = new float[inn][outn];
		this.b = new float[outn];
		this.z = new float[outn];
		this.outz = new float[outn];
		this.loss = loss;

		IntStream.range(0, outn).forEach(j->{
			this.b[j] = 0.0f;
			IntStream.range(0, inn).forEach(i->{
				this.w[i][j] = (float)(2.0 * Math.random() - 1.0) * EPS;
			});
		});
		format = OutputFormat.SIGMOID;
	}

	/**
	 * forward operation
	 * @param in
	 * @return
	 */
	public float[] forward(float[] in) {
		inz = in.clone();
		IntStream.range(0, outn).forEach(j->z[j] = 0.0f);
		IntStream.range(0, outn).parallel().forEach(j->{
			z[j] += IntStream.range(0,  inn).mapToDouble(i->in[i] * w[i][j]).sum();
			z[j] += b[j];
		});
		switch (format) {
		case SIGMOID:
			IntStream.range(0, outn).forEach(i->{
				outz[i] = sigmoid(z[i]);
			});
			break;

		case SOFTMAX:
			{
				OptionalDouble maxf = IntStream.range(0, outn).mapToDouble(i->z[i]).max();
				float softmax_div = (float) IntStream.range(0,  outn).mapToDouble(j->Math.exp(z[j] - maxf.getAsDouble())).sum();
				IntStream.range(0,  outn).forEach(i->{
					outz[i] = (float) (Math.exp(z[i] - maxf.getAsDouble()) / softmax_div);
				});
			}
			break;
		}

		return outz;
	}
}
