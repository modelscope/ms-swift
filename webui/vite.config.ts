import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    host: true,
    port: 5180,
    strictPort: false,
    // 所在机器 fs.inotify.max_user_watches 只有 8192，且已被大仓库的索引进程占满，
    // 用 inotify 会直接 ENOSPC 崩掉。webui 自身文件很少，改用轮询更稳。
    watch: {
      usePolling: true,
      interval: 400,
      ignored: ['**/node_modules/**', '**/dist/**', '**/.git/**'],
    },
  },
});
