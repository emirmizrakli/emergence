import { defineConfig } from 'vite';

// GitHub Pages serves the project under /emergence/
export default defineConfig({
  base: process.env.BASE_PATH ?? '/emergence/',
  build: {
    target: 'es2022',
    outDir: 'dist',
    sourcemap: false,
  },
  worker: {
    format: 'es',
  },
});
