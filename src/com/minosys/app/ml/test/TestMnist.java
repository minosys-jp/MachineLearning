package com.minosys.app.ml.test;

import java.io.IOException;
import java.util.Random;

import com.minosys.app.ml.lib.MNIST;
import com.minosys.app.ml.lib.NeuralNet;
import com.minosys.app.ml.lib.QuadLossFunction;

public class TestMnist {
	private static final int[] LAYERS = {768, 100, 10}; // 1 hidden layer
	private static final int BATCHSIZE = 200;
	private static final int NSET = 200;
	private static final float LRATE = 0.3f;
	private final MNIST teacher, predictor;
	private final NeuralNet nn;

	public TestMnist() throws IOException {
		this.teacher = MNIST.load_train(true, true);
		this.predictor = MNIST.load_test(true, true);
		this.nn = new NeuralNet(new QuadLossFunction(), LRATE, LAYERS);
	}

	private static int[] createSamples(MNIST mnist) {
		Random r = new Random();
		int[] samples = new int[100];
		for (int k = 0; k < samples.length; ++k) {
			samples[k] = r.nextInt(mnist.getQuantity());
		}
		return samples;
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

	private void showAccuracy(MNIST mnist, int loop, int[] samples) {
		int r = 0;
		for (int k = 0; k < samples.length; ++k) {
			float[] out = nn.forward(mnist.image.getContent(samples[k]));
			float[] label = mnist.label.getContent(samples[k]);
			if (label[argmax(out)] > 0.0f) {
				++r;
			}
		}
		double acc = (double)r / (double)samples.length * 100.0;
		System.out.printf("loop %d: accuracy=%f\n", loop, acc);
	}

	public void run() {
		for (int k = 0; k < 10; ++k) {
			int[] samples = nn.backPropagate(teacher, NSET, BATCHSIZE);
			System.out.println();
			showAccuracy(teacher, k, samples);
		}
		System.out.println("*** end of learning); now start to predict ***");
		for (int k = 0; k < 10; ++k) {
			int[] samples = createSamples(predictor);
			showAccuracy(predictor, k, samples);
		}
	}

	public static void main(String[] args) throws IOException {
		// TODO 自動生成されたメソッド・スタブ
		TestMnist test = new TestMnist();
		test.run();
	}

}
