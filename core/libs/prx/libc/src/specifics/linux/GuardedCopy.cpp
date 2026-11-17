#if defined(__linux__) && defined(__x86_64__)
#include "prx/libc/include/specifics/linux/GuardedCopy.hpp"

#include <cstdint>
#include <cstdlib>
#include <ucontext.h>

extern "C" {
bool GuestGuardedCopy(void* destination, const void* source, std::size_t bytes);
extern const char GuestGuardedCopyFault[];
}

asm(".pushsection .text\n"
    ".p2align 4\n"
    ".globl GuestGuardedCopy\n"
    ".hidden GuestGuardedCopy\n"
    ".type GuestGuardedCopy, @function\n"
    "GuestGuardedCopy:\n"
    "    cmpq $32, %rdx\n"
    "    ja 5f\n"
    "    cmpq $16, %rdx\n"
    "    jae 4f\n"
    "    cmpq $8, %rdx\n"
    "    jae 3f\n"
    "    cmpq $4, %rdx\n"
    "    jae 2f\n"
    "    testq %rdx, %rdx\n"
    "    je 6f\n"
    "    movzbl (%rsi), %eax\n"
    "    movzbl -1(%rsi,%rdx), %ecx\n"
    "    cmpq $2, %rdx\n"
    "    jbe 1f\n"
    "    movzbl 1(%rsi), %r8d\n"
    "    movb %r8b, 1(%rdi)\n"
    "1:\n"
    "    movb %al, (%rdi)\n"
    "    movb %cl, -1(%rdi,%rdx)\n"
    "    jmp 6f\n"
    "2:\n"
    "    movl (%rsi), %eax\n"
    "    movl -4(%rsi,%rdx), %ecx\n"
    "    movl %eax, (%rdi)\n"
    "    movl %ecx, -4(%rdi,%rdx)\n"
    "    jmp 6f\n"
    "3:\n"
    "    movq (%rsi), %rax\n"
    "    movq -8(%rsi,%rdx), %rcx\n"
    "    movq %rax, (%rdi)\n"
    "    movq %rcx, -8(%rdi,%rdx)\n"
    "    jmp 6f\n"
    "4:\n"
    "    movdqu (%rsi), %xmm0\n"
    "    movdqu -16(%rsi,%rdx), %xmm1\n"
    "    movdqu %xmm0, (%rdi)\n"
    "    movdqu %xmm1, -16(%rdi,%rdx)\n"
    "    jmp 6f\n"
    "5:\n"
    "    movq %rdx, %rcx\n"
    "    rep movsb\n"
    "6:\n"
    "    movl $1, %eax\n"
    "    ret\n"
    ".globl GuestGuardedCopyFault\n"
    ".hidden GuestGuardedCopyFault\n"
    "GuestGuardedCopyFault:\n"
    "    xorl %eax, %eax\n"
    "    ret\n"
    ".size GuestGuardedCopy, .-GuestGuardedCopy\n"
    ".popsection\n");

extern "C" bool GuestCopyRecoverFault_nid_no_patch(int signal, siginfo_t* info, void* context) {
    if ((signal != SIGSEGV && signal != SIGBUS) || info == nullptr || info->si_code <= 0 || context == nullptr) return false;
    auto& registers = static_cast<ucontext_t*>(context)->uc_mcontext.gregs;
    const auto rip = static_cast<std::uintptr_t>(registers[REG_RIP]);
    const auto begin = reinterpret_cast<std::uintptr_t>(&GuestGuardedCopy);
    const auto fault = reinterpret_cast<std::uintptr_t>(GuestGuardedCopyFault);
    if (rip < begin || rip >= fault) return false;
    registers[REG_RIP] = static_cast<greg_t>(fault);
    return true;
}

namespace GuardedCopy {

bool Ready() {
    static const bool ready = std::getenv("APS5_NO_DIRECT_GUEST_COPY") == nullptr && FaultsCaught();
    return ready;
}

bool Copy(void* destination, const void* source, std::size_t bytes) {
    return GuestGuardedCopy(destination, source, bytes);
}

}
#endif
