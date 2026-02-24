package daemon

import (
	"context"
	"net/http"
)

type Module interface {
	Name() string
	Init(d *Daemon) error
	RegisterRoutes(mux *http.ServeMux)
	Start(ctx context.Context) error
	Stop() error
}
