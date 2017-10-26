package com.minosys.app.ml.lib.autoencoder;

import java.io.IOException;
import java.util.stream.IntStream;

import com.minosys.app.ml.lib.ILossFunction;
import com.minosys.app.ml.lib.INoiseShaper;
import com.minosys.app.ml.lib.ImageLabelSet;
import com.minosys.app.ml.lib.NeuralNet;
import com.minosys.app.ml.lib.SimpleNet;

public class AutoEncoderNN extends NeuralNet {

	public AutoEncoderNN(ILossFunction loss, float lrate, int n1, int n2) throws IOException {
		super(loss, lrate, new int[] {n1, n2, n1});
		// TODO 自動生成されたコンストラクター・スタブ
	}

	public void copyWB(SimpleNet neuron) {
		neuron.b = this.neurons[0].b;
		neuron.w = this.neurons[0].w;
	}

	public static float[][] denoising(float[][] in, int n, INoiseShaper shaper) {
		float[][] out = new float[n][];
		if (shaper == null) {
			return in;
		}

		IntStream.range(0,  n).forEach(k->{
			out[k] = new float[in[k].length];
			IntStream.range(0, in[k].length).forEach(i->{
				out[k][i] = shaper.denoising(in[k][i])?0.0f:in[k][i];
			});
		});
		return out;
	}

	public ImageLabelSet nextLayerImageSet(ImageLabelSet ils, INoiseShaper shaper) {
		final int nils = ils.image.getQuantity();
		float[][] data = new float[nils][];
		IntStream.range(0, nils).forEach(k->{
			data[k] = this.neurons[0].forward(ils.image.getContent(k));
		});
		float[][] in = denoising(data, nils, shaper);
		return new AutoEncoderImageSet(nils, neurons[0].getOutn(), in, data);
	}

	// create AutoEncoderImageSet from ImageLabelSet
	public static AutoEncoderImageSet createImageSet(ImageLabelSet ils) {
		ImageLabelSet.BaseImage bi = ils.image;
		int wh = bi.getWidth() * bi.getHeight();
		return  new AutoEncoderImageSet(wh, bi, bi);
	}
}
