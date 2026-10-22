# Dashboard image: static Vite build served by nginx, which also reverse-
# proxies /api/* to the backend container — the same /api convention
# vite.config.js's dev proxy uses, so the browser sees one origin.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Relative base: requests go to /api/... on whatever host serves the page.
# No bearer token is baked into the bundle — nginx injects it server-side
# (DASHBOARD_AUTH=basic), or the backend's sign-in session is used (signin).
ENV VITE_API_BASE=/api
RUN npm run build

FROM nginx:1.29-alpine
RUN apk add --no-cache apache2-utils
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY docker-entrypoint.d/10-dashboard-auth.envsh /docker-entrypoint.d/10-dashboard-auth.envsh
RUN chmod +x /docker-entrypoint.d/10-dashboard-auth.envsh
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
