package com.minosys.app.ml.lib;

import java.io.BufferedInputStream;
import java.io.IOException;

/**
 * Loader インタフェース
 * @author minoru
 *
 */
public interface ILoader {
	public void loader(BufferedInputStream bis) throws IOException;
	public int getQuantity();
	public float[] getContent(int k);
	public float[][] getContent();
}
