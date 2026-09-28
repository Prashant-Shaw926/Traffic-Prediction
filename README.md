# Traffic Prediction Application

React UI plus a Flask GRU inference API. Predictions use the trained GRU model and the Kaggle hourly series (through **2017-06-30 23:00**). This is a historical simulation, not live 2026 traffic.

## Local development

Copy `.env.example` to `.env` if needed:

```
VITE_API_BASE_URL=/api
```

Vite proxies `/api` to Flask at `http://127.0.0.1:5000`. **Both terminals must be running.** If you only start `npm run dev`, the junction dropdown stays empty.

**Terminal 1 — Flask (required)**

```bash
$env:KERAS_BACKEND="torch"
.\.venv\Scripts\python.exe -m flask --app backend.app run --host 127.0.0.1 --port 5000
```

Backend: http://127.0.0.1:5000

**Terminal 2 — React**

```bash
npm install
npm run dev
```

Frontend: http://localhost:5173 (or the next free port Vite prints)

Restart `npm run dev` after changing `.env` or `vite.config.ts`.

### Demo query

- Junction 1
- 2017-06-15 10:00 to 12:00 (three hourly GRU points)

Junction 4 also has data at 2017-06-15 10:00. Dates after 30 Jun 2017 (including “today”) return an insufficient-history error from Flask. There is no mock-data fallback.

ML pipeline notes: [ml/README.md](ml/README.md). Flask API: [backend/README.md](backend/README.md).

---

# React + TypeScript + Vite


This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])

```

You can also install [eslint-plugin-react-x](https://npmx.dev/package/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://npmx.dev/package/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])

```
