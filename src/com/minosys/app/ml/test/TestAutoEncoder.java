package com.minosys.app.ml.test;

import java.io.IOException;
import java.util.stream.IntStream;

import com.minosys.app.ml.lib.MNIST;
import com.minosys.app.ml.lib.NeuralNet;
import com.minosys.app.ml.lib.QuadLossFunction;
import com.minosys.app.ml.lib.autoencoder.AutoEncoderImageSet;
import com.minosys.app.ml.lib.autoencoder.AutoEncoderNN;
import com.minosys.app.ml.lib.autoencoder.AutoEncoderTrainer;

public class TestAutoEncoder extends BaseTest {
	private static final int[] LAYERS = { 768, 100, 100, 10 };
	private static final float LRATE = 0.3f;
	private static final float THRESHOLD = 0.0f;
	private static final int NLINE = 10;
	private static final int NSET = 100;
	private static final int BATCHSIZE = 200;


	NeuralNet nn;
	MNIST mnist;

	public TestAutoEncoder() throws IOException {
		this.mnist = MNIST.load_train(true, true);
		this.nn = new NeuralNet(new QuadLossFunction(), LRATE, LAYERS);
	}

	public void run() {
		AutoEncoderTrainer aetrain = new AutoEncoderTrainer(new QuadLossFunction(), LRATE, NLINE, NSET, BATCHSIZE);
		AutoEncoderImageSet aeimage = AutoEncoderNN.createImageSet(mnist);
		aetrain.setTeacher(aeimage, null);
		aetrain.setLabels(mnist);
		recordTime();
		aetrain.train(nn);
		System.out.printf("elapsed time: %.3f sec", recordTime());
	}

	public void check(int n, int batchsize, int loop) {
		int[] samples = createRandom(batchsize, mnist.getQuantity());
		float[][] out = new float[samples.length][];
		IntStream.range(0,  batchsize).forEach(k->{
			out[k] = nn.forward(mnist.image.getContent(samples[k]));
		});
		MNIST.MNISTLabel lb = (MNIST.MNISTLabel)mnist.label;
		showAccuracy(loop, out, lb.getContent(), samples);
	}

	public void check(int n, int bs) {
		System.out.println("*** start evaluation ***");
		IntStream.range(0,  n).forEachOrdered(i->{
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
