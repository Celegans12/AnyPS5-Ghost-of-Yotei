#include "common/threadSafety.h"

namespace {

class Counter {
public:
	void Bump() {
		Common::ScopedLock lock(m_mutex);
		BumpLocked();
	}
	int Read() {
		Common::ScopedLock lock(m_mutex);
		return m_value;
	}

private:
	void BumpLocked() KYTY_REQUIRES(m_mutex) { m_value++; }

	Common::AnnotatedMutex m_mutex;
	int                    m_value KYTY_GUARDED_BY(m_mutex) = 0;
};

}

int main() {
	Counter counter;
	counter.Bump();
	return counter.Read() == 1 ? 0 : 1;
}
