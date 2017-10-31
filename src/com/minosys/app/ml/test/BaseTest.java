package com.minosys.app.ml.test;

import java.util.Random;
import java.util.stream.IntStream;

import com.minosys.app.ml.lib.INoiseShaper;

public class BaseTest {
	private Random rand = new Random();
	private long ct = 0L;

	public class PlainRandomShaper implements INoiseShaper {
		private final float threshold;

		public PlainRandomShaper(float threshold) {
			this.threshold = threshold;
		}

		public boolean denoising(float a) {
			// TODO 自動生成されたメソッド・スタブ
			return Math.abs(a - Math.random()) < threshold;
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

	protected int[] createRandom(int count, int maxn) {
		int[] y = new int[count];
		IntStream.range(0, count).forEach(k->{
			y[k] = rand.nextInt(maxn);
		});
		return y;
	}

	public void showAccuracy(int loop, float[][] out, float[][] result, int[] seq) {
		long acount = IntStream.range(0,  seq.length)
				.filter(i->result[seq[i]][argmax(out[i])] > 0.0f)
				.count();
		float perc =  (float)  acount / (float) seq.length * 100.0f;
		System.out.printf("%d: accuracy=%.3f%%\n", loop, perc);
	}

	public double recordTime() {
		long e = System.currentTimeMillis();
		double elapsed = (double)(e - ct) / 1000.0;
		ct = e;
		return elapsed;
	}
}

