package com.minosys.app.ml.test;

import java.io.IOException;
import java.util.Random;
import java.util.stream.IntStream;

import com.minosys.app.ml.lib.INoiseShaper;
import com.minosys.app.ml.lib.MNIST;
import com.minosys.app.ml.lib.NeuralNet;
import com.minosys.app.ml.lib.QuadLossFunction;
import com.minosys.app.ml.lib.autoencoder.AutoEncoderImageSet;
import com.minosys.app.ml.lib.autoencoder.AutoEncoderNN;
import com.minosys.app.ml.lib.autoencoder.AutoEncoderTrainer;

public class TestAutoEncoder {
	private static final int[] LAYERS = { 768, 100, 100, 10 };
	private static final float LRATE = 0.3f;
	private static final float THRESHOLD = 0.8f;
	private static final int NLINE = 10;
	private static final int NSET = 100;
	private static final int BATCHSIZE = 200;
	private Random rand;

	private class PlainRandomShaper implements INoiseShaper {
		private final float threshold;

		public PlainRandomShaper(float threshold) {
			this.threshold = threshold;
		}

		@Override
		public boolean denoising(float a) {
			// TODO 自動生成されたメソッド・スタブ
			return a > Math.random() - threshold;
		}

	}

	private static int argmax(float[] x) {
		float y = 0;
		int i = 0;
		for (int k = 0; k < x.length; ++k) {
			if (x[k] > y) {
				y = x[k];
				i = k;
			}
		}
		return i;
	}

	NeuralNet nn;
	MNIST mnist;

	public TestAutoEncoder() throws IOException {
		this.mnist = MNIST.load_train(true, false);
		this.nn = new NeuralNet(new QuadLossFunction(), LRATE, LAYERS);
		this.rand = new Random();
	}

	public void run() {
		AutoEncoderTrainer aetrain = new AutoEncoderTrainer(new QuadLossFunction(), LRATE, NLINE, NSET, BATCHSIZE);
		AutoEncoderImageSet aeimage = AutoEncoderNN.createImageSet(mnist);
		aetrain.setTeacher(aeimage, new PlainRandomShaper(THRESHOLD));
		aetrain.train(nn);
	}

	private int[] createRandom(int count, int maxn) {
		int[] y = new int[count];
		IntStream.range(0, count).forEach(k->{
			y[k] = rand.nextInt(maxn);
		});
		return y;
	}

	private void showAccuracy(int loop, float[][] out, byte[] result, int[] seq) {
		long acount = IntStream.range(0,  seq.length).filter(i->argmax(out[seq[i]]) == result[seq[i]]).count();
		float perc =  (float)  acount / (float) seq.length * 100.0f;
		System.out.printf("%d: accuracy=%.3f%%\n", loop, perc);
	}

	public void check(int n, int batchsize, int loop) {
		int[] samples = createRandom(batchsize, mnist.getQuantity());
		float[][] out = new float[samples.length][];
		IntStream.range(0,  batchsize).forEach(k->{
			out[k] = nn.forward(mnist.image.getContent(samples[k]));
		});
		MNIST.MNISTLabel lb = (MNIST.MNISTLabel)mnist.label;
		showAccuracy(loop, out, lb.b, samples);
	}

	public void check(int n, int bs) {
		System.out.println("*** start examination ***");
		IntStream.range(0,  bs).forEachOrdered(i->{
			check(n, bs, i);
		});
	}
	public static void main(String[] args) throws IOException {
		// TODO 自動生成されたメソッド・スタブ
		TestAutoEncoder test = new TestAutoEncoder();
		test.run();
		test.check(10, BATCHSIZE);
	}

}
