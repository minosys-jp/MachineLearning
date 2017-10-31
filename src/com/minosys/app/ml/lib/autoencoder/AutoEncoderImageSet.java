package com.minosys.app.ml.lib.autoencoder;

import java.io.BufferedInputStream;
import java.io.IOException;

import com.minosys.app.ml.lib.ILoader;
import com.minosys.app.ml.lib.INoiseShaper;
import com.minosys.app.ml.lib.ImageLabelSet;

/**
 * MNIST for AutoEncoder
 * @author minoru
 *
 */
public class AutoEncoderImageSet extends ImageLabelSet {
	public AutoEncoderImageSet(int n, int w1, int w2, float[][] images, float[][] labels) {
		super();
		image = new AutoEncoderImage(n, w1, images);
		label = new AutoEncoderLabel(n, w2, labels);
	}

	public AutoEncoderImageSet(int n, int w, float[][] images, float[][] labels) {
		this(n, w, w, images, labels);
	}

	public AutoEncoderImageSet(int w, ILoader images, ILoader labels) {
		super();
		int n = images.getQuantity();
		image = new AutoEncoderImage(n, w, images.getContent());
		label = new AutoEncoderLabel(n, w, labels.getContent());
	}

	public AutoEncoderImageSet(int w, ILoader images, ILoader labels, INoiseShaper shaper) {
		int n = images.getQuantity();
		float[][] xin = AutoEncoderNN.denoising(images.getContent(), n, shaper);
		image = new AutoEncoderImage(n, w, xin);
		label = new AutoEncoderLabel(n, w, labels.getContent());
	}

	private class AutoEncoderImage extends BaseImage {
		private final float[][] image;
		private final int n, w;

		public AutoEncoderImage(int n, int w, float[][] image) {
			this.n = n;
			this.w = w;
			this.image = image;
		}

		@Override
		public void loader(BufferedInputStream bis) throws IOException {
			// TODO 自動生成されたメソッド・スタブ
			// do nothing
		}

		@Override
		public int getQuantity() {
			// TODO 自動生成されたメソッド・スタブ
			return n;
		}

		@Override
		public float[] getContent(int k) {
			// TODO 自動生成されたメソッド・スタブ
			if (k >= 0 && k < n) {
				return image[k];
			} else {
				throw new ArrayIndexOutOfBoundsException(k);
			}
		}

		@Override
		public int getWidth() {
			// TODO 自動生成されたメソッド・スタブ
			return w;
		}

		@Override
		public int getHeight() {
			// TODO 自動生成されたメソッド・スタブ
			return 1;
		}

		@Override
		public float[][] getContent() {
			// TODO 自動生成されたメソッド・スタブ
			return image;
		}
	}

	private class AutoEncoderLabel extends BaseLabel {
		private final int n;
		private final int wh;
		private final float[][] label;

		public AutoEncoderLabel(int n, int wh, float[][] label) {
			this.n = n;
			this.wh = wh;
			this.label = label;
		}

		@Override
		public void loader(BufferedInputStream bis) throws IOException {
			// TODO 自動生成されたメソッド・スタブ
			// do nothing
		}

		@Override
		public int getQuantity() {
			// TODO 自動生成されたメソッド・スタブ
			return n;
		}

		@Override
		public float[] getContent(int k) {
			// TODO 自動生成されたメソッド・スタブ
			if (k >= 0 && k < n) {
				return label[k];
			} else {
				throw new ArrayIndexOutOfBoundsException(k);
			}
		}

		@Override
		public int getOutputCount() {
			// TODO 自動生成されたメソッド・スタブ
			return wh;
		}

		@Override
		public float[][] getContent() {
			// TODO 自動生成されたメソッド・スタブ
			return label;
		}

	}
}
