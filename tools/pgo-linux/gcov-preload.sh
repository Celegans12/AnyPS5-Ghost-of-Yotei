#!/usr/bin/env bash
# Bibliothèque partagée portant libgcov, à précharger (LD_PRELOAD) dans le jeu instrumenté : les .prx compilées avec
# -fprofile-generate référencent __gcov_indirect_call et les compteurs de libgcov, que le chargeur du jeu ne trouve pas
# sans elle (« libkernel.prx: undefined symbol: __gcov_indirect_call »). Une seule copie de libgcov pour tout le processus.
# Le jeu se termine par _exit (aucun destructeur ne tourne, donc aucun .gcda) : la bibliothèque écrit le profil
#   - dans _exit, _Exit et quick_exit (avant de rendre la main à la vraie fonction),
#   - sur SIGUSR1,
#   - toutes les APS5_GCOV_DUMP_EVERY secondes si la variable est posée (écriture puis remise à zéro des compteurs).
#   gcov-preload.sh [sortie.so]   (défaut : <arbre>/build-pgogen/libgcov-preload.so)
set -euo pipefail
dev=${APS5_DEV:-$HOME/aps5}
tree=${TREE:-src-w26}
out=${1:-$dev/$tree/build-pgogen/libgcov-preload.so}
cc=${APS5_CC:-gcc-14}
gcov=$($cc -print-file-name=libgcov.a)
[ -f "$gcov" ] || { echo "libgcov.a introuvable pour $cc" >&2; exit 1; }
src=$(mktemp --suffix=.c)
trap 'rm -f "$src"' EXIT
cat > "$src" <<'C'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>
extern void __gcov_dump(void);
extern void __gcov_reset(void);
static atomic_flag dumping = ATOMIC_FLAG_INIT;
static void dump_once(void) {
    if (!atomic_flag_test_and_set(&dumping)) { __gcov_dump(); atomic_flag_clear(&dumping); }
}
static void on_usr1(int s) { (void)s; dump_once(); }
static void* periodic(void* arg) {
    unsigned seconds = (unsigned)(unsigned long)arg;
    for (;;) {
        sleep(seconds);
        if (!atomic_flag_test_and_set(&dumping)) { __gcov_dump(); __gcov_reset(); atomic_flag_clear(&dumping); }
    }
    return 0;
}
__attribute__((constructor)) static void aps5_gcov_init(void) {
    signal(SIGUSR1, on_usr1);
    const char* every = getenv("APS5_GCOV_DUMP_EVERY");
    if (every && *every) {
        unsigned seconds = (unsigned)atoi(every);
        pthread_t thread;
        if (seconds > 0 && pthread_create(&thread, 0, periodic, (void*)(unsigned long)seconds) == 0) pthread_detach(thread);
    }
}
#define WRAP(name, type, call) \
    type name(int code) { \
        static type (*real)(int); \
        if (!real) real = (type (*)(int))dlsym(RTLD_NEXT, #name); \
        dump_once(); \
        real(code); \
        for (;;) pause(); \
    }
WRAP(_exit, void, 0)
WRAP(_Exit, void, 0)
WRAP(quick_exit, void, 0)
C
mkdir -p "$(dirname "$out")"
"$cc" -shared -fPIC -O1 -o "$out" "$src" -Wl,--whole-archive "$gcov" -Wl,--no-whole-archive -lpthread -ldl
echo "$out"
