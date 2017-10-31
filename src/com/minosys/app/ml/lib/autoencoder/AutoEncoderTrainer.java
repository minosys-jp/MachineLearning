package com.minosys.app.ml.lib.autoencoder;

import java.io.IOException;
import java.util.stream.IntStream;

import com.minosys.app.ml.lib.ILossFunction;
import com.minosys.app.ml.lib.INNTrainer;
import com.minosys.app.ml.lib.INoiseShaper;
import com.minosys.app.ml.lib.ImageLabelSet;
import com.minosys.app.ml.lib.NeuralNet;

public class AutoEncoderTrainer implements INNTrainer {
	ImageLabelSet ils, cils, lastIls;
	AutoEncoderNN ann;
	ILossFunction loss;
	int nset, batchsize, nline;
	float lrate;
	INoiseShaper shaper;

	public AutoEncoderTrainer(ILossFunction loss, float lrate, int nline, int nset, int batchsize) {
		this.loss = loss;
		this.lrate = lrate;
		this.nset = nset;
		this.nline = nline;
		this.batchsize = batchsize;
	}

	@Override
	public void setTeacher(ImageLabelSet ils, INoiseShaper shaper) {
		// TODO 自動生成されたメソッド・スタブ
		int wh = ils.image.getWidth() * ils.image.getHeight();
		int n = ils.image.getQuantity();
		this.ils = ils;
		this.shaper = shaper;
		if (shaper != null) {
			// TODO: noise shaping
			float[][] x = ils.image.getContent();
			float[][] xin = new float[n][wh];
			IntStream.range(0, n).forEach(k->{
				IntStream.range(0,  wh).forEach(j->{
					xin[k][j] = shaper.denoising(x[k][j])?0.0f:x[k][j];
				});
			});
			this.cils = new AutoEncoderImageSet(n, wh, xin, x);
		} else {
			this.cils = AutoEncoderNN.createImageSet(ils);
		}
	}

	@Override
	public void train(NeuralNet nn) {
		// TODO 自動生成されたメソッド・スタブ

		IntStream.range(0, nn.neurons.length - 1).forEachOrdered(k->{
			// correct with AutoEncoders
			AutoEncoderNN ann;
			try {
				ann = new AutoEncoderNN(loss, lrate, nn.neurons[k].getInn(), nn.neurons[k].getOutn());
				IntStream.range(0, nline).forEach(m->{
					ann.backPropagate(cils, nset, batchsize);
					System.out.print("!");
				});
				ann.copyWB(nn.neurons[k]);
				cils = ann.nextLayerImageSet(cils, shaper);
				System.out.println(">");
			} catch (IOException e) {
				// TODO 自動生成された catch ブロック
				throw new RuntimeException(e);
			}
		});

		// last level use teacher labels
		try {
			int nl = nn.neurons.length - 1;
			int[] layer = { nn.neurons[nl - 1].getInn(), nn.neurons[nl].getOutn() };
			NeuralNet nntmp = new NeuralNet(loss, lrate, layer);
			AutoEncoderImageSet ilstmp = new AutoEncoderImageSet(cils.getQuantity(), cils.image.getWidth(),
					lastIls.label.getOutputCount(), cils.image.getContent(), lastIls.label.getContent());
			IntStream.range(0,  nline).forEach(m->{
				nntmp.backPropagate(ilstmp, nset, batchsize);
			});
			nn.neurons[nl].b = nntmp.neurons[0].b;
			nn.neurons[nl].w = nntmp.neurons[0].w;
		} catch (IOException e) {
			// TODO 自動生成された catch ブロック
			new RuntimeException(e);
		}

		// fine tuning
		System.out.println("...now proceeding fine tuning");
		nn.backPropagate(ils, nset, batchsize);
	}

	@Override
	public void setLabels(ImageLabelSet lastIls) {
		// TODO 自動生成されたメソッド・スタブ
		this.lastIls = lastIls;
	}

}
