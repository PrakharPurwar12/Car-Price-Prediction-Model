package com.carintel.hadoop;

import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.RawLocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.permission.FsPermission;
import java.io.File;
import java.io.IOException;

public class WinLocalFileSystem extends LocalFileSystem {
    public WinLocalFileSystem() {
        super(new WinRawFileSystem());
    }

    public static class WinRawFileSystem extends RawLocalFileSystem {
        public WinRawFileSystem() {
            super();
        }

        @Override
        public boolean mkdirs(Path f) throws IOException {
            File p2f = pathToFile(f);
            return p2f.mkdirs() || p2f.isDirectory();
        }

        @Override
        public void setPermission(Path p, FsPermission permission) throws IOException {
            // No-op on Windows: eliminates winutils.exe entirely
        }

        @Override
        public FileStatus[] listStatus(Path f) throws IOException {
            File localf = pathToFile(f);
            if (!localf.exists()) {
                return new FileStatus[0];
            }
            if (localf.isFile()) {
                return new FileStatus[] { getFileStatus(f) };
            }
            File[] files = localf.listFiles();
            if (files == null) {
                return new FileStatus[0];
            }
            FileStatus[] results = new FileStatus[files.length];
            for (int i = 0; i < files.length; i++) {
                results[i] = getFileStatus(new Path(files[i].toURI()));
            }
            return results;
        }
    }
}
