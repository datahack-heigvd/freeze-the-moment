# The site, built for production. The data is mounted at /srv/store (`just pull-site` fills ./store).
FROM docker.io/oven/bun:1-debian
WORKDIR /srv/app
COPY app/package.json app/bun.lock app/bunfig.toml ./
RUN bun install --production --frozen-lockfile
COPY app/ ./
ENV NODE_ENV=production PORT=8000
EXPOSE 8000
CMD ["bun", "server.ts"]
