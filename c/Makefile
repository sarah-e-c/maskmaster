# maskmaster C library.
#
# Default build is hardware-free: it produces libmaskmaster.so/.a and the
# synthetic-topology smoke test, with NO dependency on libsmctrl. mm_discover()
# compiles to an ENOSYS stub.
#
# To build discovery against a real libsmctrl (requires nvdebug at runtime):
#   make DISCOVERY=1 SMCTRL_DIR=/path/to/libsmctrl
# That adds -DMM_HAVE_LIBSMCTRL, -I/-L for libsmctrl, and links -lsmctrl ONLY.
# We never link -lcuda directly (libsmctrl.so pulls it in transitively).

CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -fPIC
LDFLAGS ?=
LDLIBS  ?=

ifdef DISCOVERY
  ifndef SMCTRL_DIR
    $(error DISCOVERY=1 requires SMCTRL_DIR=/path/to/libsmctrl)
  endif
  CFLAGS  += -DMM_HAVE_LIBSMCTRL -I$(SMCTRL_DIR)
  LDFLAGS += -L$(SMCTRL_DIR)
  LDLIBS  += -lsmctrl
endif

SRC = maskmaster.c maskmaster_enum.c maskmaster_config.c
OBJ = $(SRC:.c=.o)
TESTS = test_topology test_enum

all: libmaskmaster.so libmaskmaster.a $(TESTS)

libmaskmaster.so: $(OBJ)
	$(CC) -shared -o $@ $(OBJ) $(LDFLAGS) $(LDLIBS)

libmaskmaster.a: $(OBJ)
	$(AR) rcs $@ $(OBJ)

%.o: %.c maskmaster.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Static-link the tests against the archive so they run with no shared-lib path.
test_%: test_%.c libmaskmaster.a maskmaster.h
	$(CC) $(CFLAGS) -o $@ $< libmaskmaster.a $(LDFLAGS) $(LDLIBS)

# Depend on `all` so the .so (needed by the Python binding) is always current too.
check: all
	./test_topology
	./test_enum

clean:
	rm -f $(OBJ) libmaskmaster.so libmaskmaster.a $(TESTS)

.PHONY: all check clean
