package com.minosys.app.ml.lib.autoencoder;

import java.io.IOException;
import java.util.stream.IntStream;

import com.minosys.app.ml.lib.ILossFunction;
import com.minosys.app.ml.lib.INNTrainer;
import com.minosys.app.ml.lib.INoiseShaper;
import com.minosys.app.ml.lib.ImageLabelSet;
import com.minosys.app.ml.lib.NeuralNet;

public class AutoEncoderTrainer implements INNTrainer {
	ImageLabelSet ils, cils;
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

		IntStream.range(0, nn.neurons.length).forEachOrdered(k->{
			// compensate with AutoEncoders
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

		// teacher label not used
	}

}
