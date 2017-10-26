package com.minosys.app.ml.lib;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * MNIST file objects
 * @author minoru
 *
 */
public class MNIST extends ImageLabelSet {
	private final static String train_images = "train-images-idx3-ubyte";
	private final static String train_labels = "train-labels-idx1-ubyte";
	private final static String t10k_images = "t10k-images-idx3-ubyte";
	private final static String t10k_labels = "t10k-labels-idx1-ubyte";

	/**
	 * load MNIST image file
	 * @author minoru
	 *
	 */
	public class MNISTImage extends ImageLabelSet.BaseImage {
		private  boolean normalize;
		private  int w, h, n;
		public byte b[][];
		float f[][];

		MNISTImage(boolean normalize) {
			this.normalize = normalize;
		}

		/**
		 * image file loader
		 * @param bis
		 * @throws IOException
		 */
		public void loader(BufferedInputStream bis) throws IOException {
			// magic number を無視
			MNIST.readInt(bis);;
			// 数
			n = MNIST.readInt(bis);
			// 幅
			w = MNIST.readInt(bis);
			// 高さ
			h = MNIST.readInt(bis);
			assert(w == 28 && h == 28);
			b = new byte[n][w * h];
			for (int m = 0; m < n; ++m) {
				bis.read(b[m]);
			}
			System.out.println(w);
			int[] bi = new int[w * h];
			f = new float[n][w * h];
			for (int m = 0; m < n; ++m) {
				int bmax = 0;
				for (int x = 0; x < w * h; ++x) {
					bi[x] = (int)b[m][x] & 255;
					if (bi[x] > bmax) {
						bmax = bi[x];
					}
					f[m][x] = (float)bi[x];
				}
				if (normalize) {
					for (int x = 0; x < w * h; ++x) {
						f[m][x] = f[m][x] / (float)bmax;
					}
				}
			}
		}

		@Override
		public int getQuantity() {
			// TODO 自動生成されたメソッド・スタブ
			return n;
		}

		@Override
		public int getWidth() {
			// TODO 自動生成されたメソッド・スタブ
			return w;
		}

		@Override
		public int getHeight() {
			// TODO 自動生成されたメソッド・スタブ
			return h;
		}

		@Override
		public float[] getContent(int k) {
			// TODO 自動生成されたメソッド・スタブ
			return f[k];
		}

		@Override
		public float[][] getContent() {
			return f;
		}
	}

	/**
	 * load MNIST label file
	 * @author minoru
	 *
	 */
	public class MNISTLabel extends ImageLabelSet.BaseLabel {
		public int n;
		public final boolean oneHot;
		public byte b[];
		float bhot[][];

		MNISTLabel(boolean oneHot) {
			this.oneHot = oneHot;
		}
		/**
		 * return true if two parameters are equal
		 * @param i
		 * @param j
		 * @return
		 */
		private  float getPattern(int i, int j) {
			return (i == j) ? 1.0f : 0.0f;
		}

		/**
		 * label file loader
		 * @param bis
		 * @throws IOException
		 */
		public void loader(BufferedInputStream bis) throws IOException {
			// magic number を無視
			MNIST.readInt(bis);
			// 数
			n = MNIST.readInt(bis);
			b = new byte[n];
			bis.read(b);
			if (oneHot) {
				bhot = new float[n][10];
				for (int i = 0; i < n; ++i) {
					for (int j = 0; j < 10; ++j) {
						bhot[i][j] = getPattern(b[i], j);
					}
				}
			}
		}

		@Override
		public int getQuantity() {
			// TODO 自動生成されたメソッド・スタブ
			return n;
		}

		@Override
		public float[] getContent(int k) {
			// TODO 自動生成されたメソッド・スタブ
			return bhot[k];
		}

		@Override
		public int getOutputCount() {
			// TODO 自動生成されたメソッド・スタブ
			return 10;	// 0～9
		}

		@Override
		public float[][] getContent() {
			// TODO 自動生成されたメソッド・スタブ
			return bhot;
		}
	}


	/**
	 * private constructor
	 */
	MNIST(boolean normalize, boolean oneHot) {
		image = new MNISTImage(normalize);
		label = new MNISTLabel(oneHot);
	}

	/**
	 * read 32bit integer (big endian)
	 * @param bis
	 * @return
	 * @throws IOException
	 */
	static int readInt(BufferedInputStream bis) throws IOException {
		byte[] b = new byte[4];
		bis.read(b);
		return ByteBuffer.wrap(b).getInt();
	}

	/**
	 * load the specific MNIST file set
	 *
	 * @param imagef
	 * @param labelf
	 * @param normalize
	 * @param one_hot
	 * @return
	 * @throws IOException
	 */
	public static MNIST load(final String imagef, final String labelf, boolean normalize,
			boolean oneHot) throws IOException {
		MNIST mnist = new MNIST(normalize, oneHot);
		BufferedInputStream bis = null;
		try {
			bis = new BufferedInputStream(new FileInputStream(imagef));
			mnist.image.loader(bis);
			bis.close();
			bis = new BufferedInputStream(new FileInputStream(labelf));
			mnist.label.loader(bis);
		} finally {
			if (bis != null) bis.close();
		}
		return mnist;
	}

	/**
	 * load 60000 samples for training
	 *
	 * @param normalize
	 * @param one_hot
	 * @return
	 * @throws IOException
	 */
	public static MNIST load_train(boolean normalize, boolean one_hot) throws IOException{
		return load(train_images, train_labels, normalize, one_hot);
	}

	/**
	 * load 10000 samples for test
	 *
	 * @param normalize
	 * @param one_hot
	 * @return
	 * @throws IOException
	 */
	public static MNIST load_test(boolean normalize, boolean one_hot) throws IOException{
		return load(t10k_images, t10k_labels, normalize, one_hot);
	}

	public static void main(String[] args) throws IOException {
		MNIST train = MNIST.load_train(true, false);
		MNIST test = MNIST.load_test(true, false);
		System.out.printf("train: %d samples; %d width %d height",
				train.image.getQuantity(), train.image.getWidth(), train.image.getHeight());
		System.out.printf(" %d labels:",
				train.label.getQuantity());
		System.out.printf("test: %d samples; %d width %d height",
				test.image.getQuantity(), test.image.getWidth(), test.image.getHeight());
		System.out.printf(" %d labels:",
				test.label.getQuantity());
	}
}
