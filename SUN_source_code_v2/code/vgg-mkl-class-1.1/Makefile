# file: Makefile
# author: Andrea Vedaldi

VER  := 1.1
VGG_MKL_CLASS_DIST := vgg-mkl-class-$(VER)

GIT := git

.PHONY: clean, all, all-mex

all: all-mex

clean:
	rm -rf vgg-mkl-class
	rm -f `find . -name '*~'`
	rm -f `find . -name '#*#'`
	rm -f `find . -name '.#*#'`
	rm -f `find . -name '._*'`
	rm -f `find . -name '.fuse_*'`
	rm -f `find . -name 'core'`
	rm -f `find . -name '*.map'`
	rm -f `find . -name '*.manifest'`
	rm -f `find . -name '*.pdb'`
	rm -f `find . -name '*.obj'`
	rm -f `find . -name '*.o'`
	rm -f `find . -name '*.ilk'`

dist-clean: clean
	rm -f `find . -name '*.mex*'`

dist-class: clean
	rm -rf $(VGG_MKL_CLASS_DIST) $(VGG_MKL_CLASS_DIST)-tmp
	mkdir $(VGG_MKL_CLASS_DIST)
	$(GIT) archive --prefix=$(VGG_MKL_CLASS_DIST)-tmp/ bin/v$(VER)-bin | tar xv
	rsync -avz --include-from=vgg-mkl-class.txt --exclude='*' $(VGG_MKL_CLASS_DIST)-tmp/ $(VGG_MKL_CLASS_DIST)
	tar zcf $(VGG_MKL_CLASS_DIST).tar.gz $(VGG_MKL_CLASS_DIST)
	rm -rf $(VGG_MKL_CLASS_DIST) $(VGG_MKL_CLASS_DIST)-tmp

# --------------------------------------------------------------------
#                                   Determine architecture and compile
# --------------------------------------------------------------------

Darwin_PPC_ARCH             := mac
Darwin_Power_Macintosh_ARCH := mac
Darwin_i386_ARCH            := maci
Linux_i386_ARCH             := glx
Linux_i686_ARCH             := glx
Linux_unknown_ARCH          := glx
Linux_x86_64_ARCH           := a64

UNAME := $(shell uname -sm)
ARCH  ?= $($(shell echo "$(UNAME)" | tr \  _)_ARCH)

# sanity check
ifeq ($(ARCH),)
die:=$(error $(err_no_arch))
endif

all-mex:
	matlab -$(ARCH) -nojvm -r 'compile;exit;'

# --------------------------------------------------------------------
#                                                    Make distribution
# --------------------------------------------------------------------

.PHONY: bin-release, bin-commit, bin-dist, src-dist

bin-release:
	echo Fetching remote tags ; \
	$(GIT) fetch --tags ; \
	echo Checking out v$(VER) ; \
	$(GIT) checkout v$(VER)
	echo Rebuilding binaries for release ; \
	make all

bin-commit: bin-release
	@set -e ; \
	echo Fetching remote tags ; \
	$(GIT) fetch --tags ; \
	BRANCH=v$(VER)-$(ARCH)  ; \
	echo Crearing/resetting and checking out branch $$BRANCH to v$(VER); \
	$(GIT) branch -f $$BRANCH v$(VER) ; \
	$(GIT) checkout $$BRANCH ; \
	echo Adding binaries ; \
	$(GIT) add -f $(shell find . -name '*.mex$(ARCH)') ; \
	if test -z "$$($(GIT) diff --cached)" ; \
	then \
	  echo No changes to commit ; \
	else \
	  echo Commiting changes ; \
	  $(GIT) commit -m "$(ARCH) binaries for version $(VER)" ; \
	fi ; \
	echo Commiting and pushing to server the binaries ; \
	$(GIT) push -v --force bin $$BRANCH:$$BRANCH ; \
	$(GIT) checkout v$(VER) ; \
	$(GIT) branch -D $$BRANCH ;

bin-merge:
	echo Fetching remote tags ; \
	$(GIT) fetch --tags ; \
	set -e ; \
	echo Checking out $(VER) ; \
	$(GIT) checkout v$(VER) ; \
	BRANCH=v$(VER)-bin ; \
	echo Crearing/resetting and checking out branch $$BRANCH to v$(VER); \
	$(GIT) branch -f $$BRANCH v$(VER) ; \
	$(GIT) checkout $$BRANCH ; \
	MERGE_BRANCHES=; \
	for ALT_ARCH in maci maci64 glx a64 w32 w64 ; \
	do \
	  MERGE_BRANCH=v$(VER)-$$ALT_ARCH ; \
	  echo Fetching $$MERGE_BRANCH ; \
	  $(GIT) fetch -f bin $$MERGE_BRANCH:remotes/bin/$$MERGE_BRANCH ; \
	  MERGE_BRANCHES="$$MERGE_BRANCHES bin/$$MERGE_BRANCH" ; \
	done ; \
	echo merging $$MERGE_BRANCHES ; \
	$(GIT) merge -m "Merged binaries $$MERGE_BRANCHES" $$MERGE_BRANCHES ; \
	echo Pushing to server the merged binaries ; \
	$(GIT) push -v --force bin $$BRANCH:$$BRANCH ; \
	$(GIT) checkout v$(VER) ; \
	$(GIT) branch -D $$BRANCH ;
