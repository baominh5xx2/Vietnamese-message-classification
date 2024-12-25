import pandas as pd

# Danh sách các nhận xét sản phẩm và nhãn
reviews = [
    ("Sản phẩm rất tốt, tôi rất hài lòng.", 1),
    ("Chất lượng vượt mong đợi!", 1),
    ("Hàng giao nhanh, đóng gói cẩn thận.", 1),
    ("Không thể tin được sản phẩm lại tốt đến thế!", 1),
    ("Tôi sẽ giới thiệu cho bạn bè về sản phẩm này.", 1),
    ("Dịch vụ khách hàng rất tuyệt vời.", 1),
    ("Giá cả hợp lý và chất lượng cao.", 1),
    ("Sản phẩm hoàn toàn phù hợp với mô tả.", 1),
    ("Tôi rất thích thiết kế của sản phẩm.", 1),
    ("Cảm ơn shop đã mang đến sản phẩm tốt như vậy!", 1),
    ("Hàng không như mô tả, rất thất vọng.", 0),
    ("Sản phẩm kém chất lượng, không đáng tiền.", 0),
    ("Giao hàng chậm trễ hơn dự kiến.", 0),
    ("Đóng gói không cẩn thận, sản phẩm bị hư hại.", 0),
    ("Hỗ trợ khách hàng không tốt, tôi không hài lòng.", 0),
    ("Giá quá cao so với chất lượng.", 0),
    ("Sản phẩm không hoạt động như quảng cáo.", 0),
    ("Thiết kế không đẹp như hình ảnh.", 0),
    ("Tôi sẽ không mua lại sản phẩm này.", 0),
    ("Rất tiếc vì đã mua sản phẩm này.", 0),
    ("Màu sắc sản phẩm rất đẹp và bắt mắt.", 1),
    ("Sản phẩm dễ sử dụng và hiệu quả.", 1),
    ("Hàng chất lượng, tôi rất hài lòng.", 1),
    ("Thiết kế hiện đại, rất phù hợp với nhu cầu.", 1),
    ("Shop tư vấn nhiệt tình, giao hàng đúng hẹn.", 1),
    ("Đây là sản phẩm tốt nhất tôi từng mua.", 1),
    ("Chất liệu sản phẩm rất bền và đẹp.", 1),
    ("Sản phẩm thật sự đáng giá từng đồng.", 1),
    ("Dịch vụ sau bán hàng rất tốt.", 1),
    ("Hàng giao đúng như quảng cáo, rất hài lòng.", 1),
    ("Sản phẩm có nhiều tính năng vượt trội.", 1),
    ("Shop làm ăn uy tín, sẽ ủng hộ tiếp.", 1),
    ("Sản phẩm này không hữu ích như tôi nghĩ.", 0),
    ("Hàng giao thiếu phụ kiện, không hài lòng.", 0),
    ("Sản phẩm có lỗi, không sử dụng được.", 0),
    ("Tôi đã nhận hàng sai màu, rất bực mình.", 0),
    ("Đóng gói sơ sài, hàng bị trầy xước.", 0),
    ("Giao hàng quá lâu, không đáng chờ.", 0),
    ("Sản phẩm không giống với hình trên web.", 0),
    ("Shop không hỗ trợ đổi trả, rất thất vọng.", 0),
    ("Chất lượng sản phẩm không ổn định.", 0),
    ("Dịch vụ khách hàng rất tệ.", 0),
    ("Sản phẩm này đã thay đổi cuộc sống của tôi.", 1),
    ("Tôi cảm thấy rất hài lòng với chất lượng.", 1),
    ("Đây là món quà tuyệt vời cho gia đình.", 1),
    ("Tôi sẽ mua lại sản phẩm này lần sau.", 1),
    ("Sản phẩm xứng đáng nhận 5 sao.", 1),
    ("Hàng chính hãng, chất lượng tuyệt vời.", 1),
    ("Thiết kế nhỏ gọn, dễ mang theo.", 1),
    ("Hàng giao nhanh, tôi rất vui.", 1),
    ("Shop hỗ trợ giải đáp thắc mắc rất tốt.", 1),
    ("Chất lượng vượt mong đợi, cảm ơn shop!", 1),
    ("Sản phẩm bị hư hỏng ngay khi nhận hàng.", 0),
    ("Hàng giao nhầm, tôi rất thất vọng.", 0),
    ("Sản phẩm không đáng giá tiền bỏ ra.", 0),
    ("Hỗ trợ khách hàng quá kém, không hài lòng.", 0),
    ("Chất lượng không như mong đợi.", 0),
    ("Đóng gói không kỹ, sản phẩm bị vỡ.", 0),
    ("Tôi không thể sử dụng sản phẩm này.", 0),
    ("Sản phẩm quá đắt so với chất lượng.", 0),
    ("Dịch vụ kém, tôi rất thất vọng.", 0),
    ("Không đáng để mua, quá thất vọng.", 0),
    ("Mẫu mã rất đẹp, phù hợp với nhu cầu.", 1),
    ("Tôi rất ấn tượng với chất lượng sản phẩm.", 1),
    ("Sản phẩm dễ dàng lắp đặt và sử dụng.", 1),
    ("Dịch vụ nhanh chóng, rất chuyên nghiệp.", 1),
    ("Sản phẩm đúng như kỳ vọng của tôi.", 1),
    ("Tôi rất thích sản phẩm này.", 1),
    ("Sản phẩm đáng giá, sẽ mua thêm.", 1),
    ("Shop giao hàng đúng hẹn, rất hài lòng.", 1),
    ("Chất lượng sản phẩm không có gì để chê.", 1),
    ("Đây là sản phẩm mà tôi cần từ lâu.", 1),
    ("Sản phẩm này có mùi khó chịu, không thể sử dụng.", 0),
    ("Giao hàng không đúng địa chỉ, rất bất tiện.", 0),
    ("Sản phẩm dễ vỡ, không bền.", 0),
    ("Hỗ trợ đổi trả chậm chạp, không hài lòng.", 0),
    ("Tôi nhận được sản phẩm đã qua sử dụng.", 0),
    ("Sản phẩm không tương thích với thiết bị của tôi.", 0),
    ("Không đáng để đầu tư, rất thất vọng.", 0),
    ("Thiết kế lỗi thời, không hấp dẫn.", 0),
    ("Tôi sẽ không bao giờ mua lại từ shop này.", 0),
    ("Sản phẩm gây phiền toái hơn là tiện ích.", 0),
    ("Màu sắc tươi sáng, rất phù hợp với sở thích của tôi.", 1),
    ("Thiết kế đẹp mắt, khiến tôi rất hài lòng.", 1),
    ("Chất liệu thân thiện với môi trường, rất tốt.", 1),
    ("Tôi đã giới thiệu sản phẩm này cho bạn bè.", 1),
    ("Sản phẩm này thực sự mang lại giá trị.", 1),
    ("Dịch vụ khách hàng hỗ trợ rất nhiệt tình.", 1),
    ("Hàng giao nhanh hơn dự kiến, rất thích.", 1),
    ("Sản phẩm bền, tôi đã sử dụng hơn 1 năm.", 1),
    ("Shop có chính sách bảo hành rất tốt.", 1),
    ("Sản phẩm nhỏ gọn, tiện lợi mang theo.", 1),
    ("Chất lượng sản phẩm làm tôi rất ngạc nhiên.", 1),
    ("Hàng giao không đủ số lượng, rất thất vọng.", 0),
    ("Sản phẩm bị méo mó, không sử dụng được.", 0),
    ("Shop không phản hồi khi tôi cần hỗ trợ.", 0),
    ("Sản phẩm không đáng tin cậy.", 0),
    ("Hàng quá hạn sử dụng, không thể dùng.", 0),
    ("Tôi cảm thấy rất thất vọng với dịch vụ.", 0),
    ("Sản phẩm không có tính năng như quảng cáo.", 0),
    ("Giao hàng quá lâu, tôi không hài lòng.", 0),
    ("Tôi đã yêu cầu trả lại sản phẩm này.", 0),
    ("Sản phẩm làm từ chất liệu kém chất lượng.", 0),
    ("Đây là món quà tuyệt vời mà tôi dành cho người thân.", 1),
    ("Hàng giao đúng như cam kết, tôi rất vui.", 1),
    ("Thiết kế sáng tạo, rất độc đáo.", 1),
    ("Chất lượng vượt xa mong đợi, tuyệt vời!", 1),
    ("Shop hỗ trợ bảo hành nhanh chóng, rất hài lòng.", 1),
    ("Sản phẩm này đáng giá đến từng đồng.", 1),
    ("Dịch vụ rất chuyên nghiệp, giao hàng nhanh.", 1),
    ("Tôi đã giới thiệu sản phẩm này cho cả gia đình.", 1),
    ("Mua hàng từ shop này rất đáng tin cậy.", 1),
    ("Hàng chính hãng, chất lượng như quảng cáo.", 1),
    ("Tôi sẽ tiếp tục mua thêm sản phẩm từ shop.", 1),
    ("Sản phẩm quá to, không tiện lợi.", 0),
    ("Thiết kế không vừa ý, tôi rất tiếc.", 0),
    ("Chất lượng sản phẩm không xứng với giá tiền.", 0),
    ("Shop giao nhầm sản phẩm, không chuyên nghiệp.", 0),
    ("Sản phẩm gây nhiều phiền phức khi sử dụng.", 0),
    ("Không thể lắp ráp sản phẩm, rất bực mình.", 0),
    ("Giao hàng sai thời gian, không đúng hẹn.", 0),
    ("Chất lượng sản phẩm làm tôi thất vọng.", 0),
    ("Shop không hỗ trợ trả hàng, rất tệ.", 0),
    ("Tôi sẽ không mua lại từ shop này nữa.", 0),
    ("Sản phẩm rất tốt, tôi rất hài lòng.", 1),
    ("Chất lượng vượt mong đợi!", 1),
    ("Hàng giao nhanh, đóng gói cẩn thận.", 1),
    ("Không thể tin được sản phẩm lại tốt đến thế!", 1),
    ("Tôi sẽ giới thiệu cho bạn bè về sản phẩm này.", 1),
    ("Dịch vụ khách hàng rất tuyệt vời.", 1),
    ("Giá cả hợp lý và chất lượng cao.", 1),
    ("Sản phẩm hoàn toàn phù hợp với mô tả.", 1),
    ("Tôi rất thích thiết kế của sản phẩm.", 1),
    ("Cảm ơn shop đã mang đến sản phẩm tốt như vậy!", 1),
    ("Hàng không như mô tả, rất thất vọng.", 0),
    ("Sản phẩm kém chất lượng, không đáng tiền.", 0),
    ("Giao hàng chậm trễ hơn dự kiến.", 0),
    ("Đóng gói không cẩn thận, sản phẩm bị hư hại.", 0),
    ("Hỗ trợ khách hàng không tốt, tôi không hài lòng.", 0),
    ("Giá quá cao so với chất lượng.", 0),
    ("Sản phẩm không hoạt động như quảng cáo.", 0),
    ("Thiết kế không đẹp như hình ảnh.", 0),
    ("Tôi sẽ không mua lại sản phẩm này.", 0),
    ("Rất tiếc vì đã mua sản phẩm này.", 0),
    ("Màu sắc sản phẩm rất đẹp và bắt mắt.", 1),
    ("Sản phẩm dễ sử dụng và hiệu quả.", 1),
    ("Hàng chất lượng, tôi rất hài lòng.", 1),
    ("Thiết kế hiện đại, rất phù hợp với nhu cầu.", 1),
    ("Shop tư vấn nhiệt tình, giao hàng đúng hẹn.", 1),
    ("Đây là sản phẩm tốt nhất tôi từng mua.", 1),
    ("Chất liệu sản phẩm rất bền và đẹp.", 1),
    ("Sản phẩm thật sự đáng giá từng đồng.", 1),
    ("Dịch vụ sau bán hàng rất tốt.", 1),
    ("Hàng giao đúng như quảng cáo, rất hài lòng.", 1),
    ("Sản phẩm có nhiều tính năng vượt trội.", 1),
    ("Shop làm ăn uy tín, sẽ ủng hộ tiếp.", 1),
    ("Sản phẩm này không hữu ích như tôi nghĩ.", 0),
    ("Hàng giao thiếu phụ kiện, không hài lòng.", 0),
    ("Sản phẩm có lỗi, không sử dụng được.", 0),
    ("Tôi đã nhận hàng sai màu, rất bực mình.", 0),
    ("Đóng gói sơ sài, hàng bị trầy xước.", 0),
    ("Giao hàng quá lâu, không đáng chờ.", 0),
    ("Sản phẩm không giống với hình trên web.", 0),
    ("Shop không hỗ trợ đổi trả, rất thất vọng.", 0),
    ("Chất lượng sản phẩm không ổn định.", 0),
    ("Dịch vụ khách hàng rất tệ.", 0),
    ("Sản phẩm này đã thay đổi cuộc sống của tôi.", 1),
    ("Tôi cảm thấy rất hài lòng với chất lượng.", 1),
    ("Đây là món quà tuyệt vời cho gia đình.", 1),
    ("Tôi sẽ mua lại sản phẩm này lần sau.", 1),
    ("Sản phẩm xứng đáng nhận 5 sao.", 1),
    ("Hàng chính hãng, chất lượng tuyệt vời.", 1),
    ("Thiết kế nhỏ gọn, dễ mang theo.", 1),
    ("Hàng giao nhanh, tôi rất vui.", 1),
    ("Shop hỗ trợ giải đáp thắc mắc rất tốt.", 1),
    ("Chất lượng vượt mong đợi, cảm ơn shop!", 1),
    ("Sản phẩm bị hư hỏng ngay khi nhận hàng.", 0),
    ("Hàng giao nhầm, tôi rất thất vọng.", 0),
    ("Sản phẩm không đáng giá tiền bỏ ra.", 0),
    ("Hỗ trợ khách hàng quá kém, không hài lòng.", 0),
    ("Chất lượng không như mong đợi.", 0),
    ("Đóng gói không kỹ, sản phẩm bị vỡ.", 0),
    ("Tôi không thể sử dụng sản phẩm này.", 0),
    ("Sản phẩm quá đắt so với chất lượng.", 0),
    ("Dịch vụ kém, tôi rất thất vọng.", 0),
    ("Không đáng để mua, quá thất vọng.", 0),
    ("Mẫu mã rất đẹp, phù hợp với nhu cầu.", 1),
    ("Tôi rất ấn tượng với chất lượng sản phẩm.", 1),
    ("Sản phẩm dễ dàng lắp đặt và sử dụng.", 1),
    ("Dịch vụ nhanh chóng, rất chuyên nghiệp.", 1),
    ("Sản phẩm đúng như kỳ vọng của tôi.", 1),
    ("Tôi rất thích sản phẩm này.", 1),
    ("Sản phẩm đáng giá, sẽ mua thêm.", 1),
    ("Shop giao hàng đúng hẹn, rất hài lòng.", 1),
    ("Chất lượng sản phẩm không có gì để chê.", 1),
    ("Đây là sản phẩm mà tôi cần từ lâu.", 1),
    ("Sản phẩm này có mùi khó chịu, không thể sử dụng.", 0),
    ("Giao hàng không đúng địa chỉ, rất bất tiện.", 0),
    ("Sản phẩm dễ vỡ, không bền.", 0),
    ("Hỗ trợ đổi trả chậm chạp, không hài lòng.", 0),
    ("Tôi nhận được sản phẩm đã qua sử dụng.", 0),
    ("Sản phẩm không tương thích với thiết bị của tôi.", 0),
    ("Không đáng để đầu tư, rất thất vọng.", 0),
    ("Thiết kế lỗi thời, không hấp dẫn.", 0),
    ("Tôi sẽ không bao giờ mua lại từ shop này.", 0),
    ("Sản phẩm gây phiền toái hơn là tiện ích.", 0),
    ("Màu sắc tươi sáng, rất phù hợp với sở thích của tôi.", 1),
    ("Thiết kế đẹp mắt, khiến tôi rất hài lòng.", 1),
    ("Chất liệu thân thiện với môi trường, rất tốt.", 1),
    ("Tôi đã giới thiệu sản phẩm này cho bạn bè.", 1),
    ("Sản phẩm này thực sự mang lại giá trị.", 1),
    ("Dịch vụ khách hàng hỗ trợ rất nhiệt tình.", 1),
    ("Hàng giao nhanh hơn dự kiến, rất thích.", 1),
    ("Sản phẩm bền, tôi đã sử dụng hơn 1 năm.", 1),
    ("Shop có chính sách bảo hành rất tốt.", 1),
    ("Sản phẩm nhỏ gọn, tiện lợi mang theo.", 1),
    ("Chất lượng sản phẩm làm tôi rất ngạc nhiên.", 1),
    ("Hàng giao không đủ số lượng, rất thất vọng.", 0),
    ("Sản phẩm bị méo mó, không sử dụng được.", 0),
    ("Shop không phản hồi khi tôi cần hỗ trợ.", 0),
    ("Sản phẩm không đáng tin cậy.", 0),
    ("Hàng quá hạn sử dụng, không thể dùng.", 0),
    ("Tôi cảm thấy rất thất vọng với dịch vụ.", 0),
    ("Sản phẩm không có tính năng như quảng cáo.", 0),
    ("Giao hàng quá lâu, tôi không hài lòng.", 0),
    ("Tôi đã yêu cầu trả lại sản phẩm này.", 0),
    ("Sản phẩm làm từ chất liệu kém chất lượng.", 0),
    ("Đây là món quà tuyệt vời mà tôi dành cho người thân.", 1),
    ("Hàng giao đúng như cam kết, tôi rất vui.", 1),
    ("Thiết kế sáng tạo, rất độc đáo.", 1),
    ("Chất lượng vượt xa mong đợi, tuyệt vời!", 1),
    ("Shop hỗ trợ bảo hành nhanh chóng, rất hài lòng.", 1),
    ("Sản phẩm này đáng giá đến từng đồng.", 1),
    ("Dịch vụ rất chuyên nghiệp, giao hàng nhanh.", 1),
    ("Tôi đã giới thiệu sản phẩm này cho cả gia đình.", 1),
    ("Mua hàng từ shop này rất đáng tin cậy.", 1),
    ("Hàng chính hãng, chất lượng như quảng cáo.", 1),
    ("Tôi sẽ tiếp tục mua thêm sản phẩm từ shop.", 1),
    ("Sản phẩm quá to, không tiện lợi.", 0),
    ("Thiết kế không vừa ý, tôi rất tiếc.", 0),
    ("Chất lượng sản phẩm không xứng với giá tiền.", 0),
    ("Shop giao nhầm sản phẩm, không chuyên nghiệp.", 0),
    ("Sản phẩm gây nhiều phiền phức khi sử dụng.", 0),
    ("Không thể lắp ráp sản phẩm, rất bực mình.", 0),
    ("Giao hàng sai thời gian, không đúng hẹn.", 0),
    ("Chất lượng sản phẩm làm tôi thất vọng.", 0),
    ("Shop không hỗ trợ trả hàng, rất tệ.", 0),
    ("Tôi sẽ không mua lại từ shop này nữa.", 0),
    ("Sản phẩm có mùi thơm dễ chịu, rất hài lòng.", 1),
    ("Thiết kế độc đáo, chưa từng thấy ở đâu.", 1),
    ("Chất lượng cao cấp, hoàn toàn yên tâm.", 1),
    ("Tôi rất thích cách đóng gói sản phẩm.", 1),
    ("Hàng giao đủ số lượng, đúng như yêu cầu.", 1),
    ("Sản phẩm đẹp hơn mong đợi của tôi.", 1),
    ("Mọi người trong gia đình đều yêu thích sản phẩm này.", 1),
    ("Dịch vụ của shop rất chuyên nghiệp và tận tình.", 1),
    ("Tôi cảm thấy rất hạnh phúc khi sử dụng sản phẩm này.", 1),
    ("Đây là món quà hoàn hảo cho người thân.", 1),
    ("Sản phẩm gây nhiều khó khăn khi sử dụng.", 0),
    ("Tôi không nhận được phụ kiện đi kèm.", 0),
    ("Shop giao nhầm màu sản phẩm, rất thất vọng.", 0),
    ("Hỗ trợ không nhiệt tình, cảm giác rất khó chịu.", 0),
    ("Sản phẩm kém chất lượng, không bền.", 0),
    ("Dịch vụ bảo hành rất chậm, không chuyên nghiệp.", 0),
    ("Tôi đã yêu cầu đổi trả nhưng chưa được xử lý.", 0),
    ("Shop không đáng tin cậy, sẽ không mua lại.", 0),
    ("Hàng không có tem chính hãng, rất nghi ngờ.", 0),
    ("Sản phẩm không đáp ứng được kỳ vọng của tôi.", 0),
    ("Tôi cảm thấy đây là một quyết định mua hàng đúng đắn.", 1),
    ("Chất lượng dịch vụ của shop thật tuyệt vời.", 1),
    ("Hàng giao nhanh, đóng gói rất cẩn thận.", 1),
    ("Sản phẩm rất đẹp, phù hợp với phong cách của tôi.", 1),
    ("Giá cả hợp lý, chất lượng vượt xa mong đợi.", 1),
    ("Tôi sẽ tiếp tục ủng hộ shop trong tương lai.", 1),
    ("Sản phẩm có hướng dẫn sử dụng rất chi tiết.", 1),
    ("Tôi rất hài lòng với sự nhiệt tình của shop.", 1),
    ("Sản phẩm đạt chuẩn chất lượng cao.", 1),
    ("Shop làm việc rất chuyên nghiệp và uy tín.", 1),
    ("Sản phẩm rất tốt, tôi rất hài lòng.", 1),
    ("Chất lượng vượt mong đợi!", 1),
    ("Hàng giao nhanh, đóng gói cẩn thận.", 1),
    ("Không thể tin được sản phẩm lại tốt đến thế!", 1),
    ("Tôi sẽ giới thiệu cho bạn bè về sản phẩm này.", 1),
    ("Dịch vụ khách hàng rất tuyệt vời.", 1),
    ("Giá cả hợp lý và chất lượng cao.", 1),
    ("Sản phẩm hoàn toàn phù hợp với mô tả.", 1),
    ("Tôi rất thích thiết kế của sản phẩm.", 1),
    ("Cảm ơn shop đã mang đến sản phẩm tốt như vậy!", 1),
    ("Hàng không như mô tả, rất thất vọng.", 0),
    ("Sản phẩm kém chất lượng, không đáng tiền.", 0),
    ("Giao hàng chậm trễ hơn dự kiến.", 0),
    ("Đóng gói không cẩn thận, sản phẩm bị hư hại.", 0),
    ("Hỗ trợ khách hàng không tốt, tôi không hài lòng.", 0),
    ("Giá quá cao so với chất lượng.", 0),
    ("Sản phẩm không hoạt động như quảng cáo.", 0),
    ("Thiết kế không đẹp như hình ảnh.", 0),
    ("Tôi sẽ không mua lại sản phẩm này.", 0),
    ("Rất tiếc vì đã mua sản phẩm này.", 0),
    ("Màu sắc sản phẩm rất đẹp và bắt mắt.", 1),
    ("Sản phẩm dễ sử dụng và hiệu quả.", 1),
    ("Hàng chất lượng, tôi rất hài lòng.", 1),
    ("Thiết kế hiện đại, rất phù hợp với nhu cầu.", 1),
    ("Shop tư vấn nhiệt tình, giao hàng đúng hẹn.", 1),
    ("Đây là sản phẩm tốt nhất tôi từng mua.", 1),
    ("Chất liệu sản phẩm rất bền và đẹp.", 1),
    ("Sản phẩm thật sự đáng giá từng đồng.", 1),
    ("Dịch vụ sau bán hàng rất tốt.", 1),
    ("Hàng giao đúng như quảng cáo, rất hài lòng.", 1),
    ("Sản phẩm có nhiều tính năng vượt trội.", 1),
    ("Shop làm ăn uy tín, sẽ ủng hộ tiếp.", 1),
    ("Sản phẩm này không hữu ích như tôi nghĩ.", 0),
    ("Hàng giao thiếu phụ kiện, không hài lòng.", 0),
    ("Sản phẩm có lỗi, không sử dụng được.", 0),
    ("Tôi đã nhận hàng sai màu, rất bực mình.", 0),
    ("Đóng gói sơ sài, hàng bị trầy xước.", 0),
    ("Giao hàng quá lâu, không đáng chờ.", 0),
    ("Sản phẩm không giống với hình trên web.", 0),
    ("Shop không hỗ trợ đổi trả, rất thất vọng.", 0),
    ("Chất lượng sản phẩm không ổn định.", 0),
    ("Dịch vụ khách hàng rất tệ.", 0),
    ("Sản phẩm này đã thay đổi cuộc sống của tôi.", 1),
    ("Tôi cảm thấy rất hài lòng với chất lượng.", 1),
    ("Đây là món quà tuyệt vời cho gia đình.", 1),
    ("Tôi sẽ mua lại sản phẩm này lần sau.", 1),
    ("Sản phẩm xứng đáng nhận 5 sao.", 1),
    ("Hàng chính hãng, chất lượng tuyệt vời.", 1),
    ("Thiết kế nhỏ gọn, dễ mang theo.", 1),
    ("Hàng giao nhanh, tôi rất vui.", 1),
    ("Shop hỗ trợ giải đáp thắc mắc rất tốt.", 1),
    ("Chất lượng vượt mong đợi, cảm ơn shop!", 1),
    ("Sản phẩm bị hư hỏng ngay khi nhận hàng.", 0),
    ("Hàng giao nhầm, tôi rất thất vọng.", 0),
    ("Sản phẩm không đáng giá tiền bỏ ra.", 0),
    ("Hỗ trợ khách hàng quá kém, không hài lòng.", 0),
    ("Chất lượng không như mong đợi.", 0),
    ("Đóng gói không kỹ, sản phẩm bị vỡ.", 0),
    ("Tôi không thể sử dụng sản phẩm này.", 0),
    ("Sản phẩm quá đắt so với chất lượng.", 0),
    ("Dịch vụ kém, tôi rất thất vọng.", 0),
    ("Không đáng để mua, quá thất vọng.", 0),
    ("Mẫu mã rất đẹp, phù hợp với nhu cầu.", 1),
    ("Tôi rất ấn tượng với chất lượng sản phẩm.", 1),
    ("Sản phẩm dễ dàng lắp đặt và sử dụng.", 1),
    ("Dịch vụ nhanh chóng, rất chuyên nghiệp.", 1),
    ("Sản phẩm đúng như kỳ vọng của tôi.", 1),
    ("Tôi rất thích sản phẩm này.", 1),
    ("Sản phẩm đáng giá, sẽ mua thêm.", 1),
    ("Shop giao hàng đúng hẹn, rất hài lòng.", 1),
    ("Chất lượng sản phẩm không có gì để chê.", 1),
    ("Đây là sản phẩm mà tôi cần từ lâu.", 1),
    ("Sản phẩm này có mùi khó chịu, không thể sử dụng.", 0),
    ("Giao hàng không đúng địa chỉ, rất bất tiện.", 0),
    ("Sản phẩm dễ vỡ, không bền.", 0),
    ("Hỗ trợ đổi trả chậm chạp, không hài lòng.", 0),
    ("Tôi nhận được sản phẩm đã qua sử dụng.", 0),
    ("Sản phẩm không tương thích với thiết bị của tôi.", 0),
    ("Không đáng để đầu tư, rất thất vọng.", 0),
    ("Thiết kế lỗi thời, không hấp dẫn.", 0),
    ("Tôi sẽ không bao giờ mua lại từ shop này.", 0),
    ("Sản phẩm gây phiền toái hơn là tiện ích.", 0),
    ("Màu sắc tươi sáng, rất phù hợp với sở thích của tôi.", 1),
    ("Thiết kế đẹp mắt, khiến tôi rất hài lòng.", 1),
    ("Chất liệu thân thiện với môi trường, rất tốt.", 1),
    ("Tôi đã giới thiệu sản phẩm này cho bạn bè.", 1),
    ("Sản phẩm này thực sự mang lại giá trị.", 1),
    ("Dịch vụ khách hàng hỗ trợ rất nhiệt tình.", 1),
    ("Hàng giao nhanh hơn dự kiến, rất thích.", 1),
    ("Sản phẩm bền, tôi đã sử dụng hơn 1 năm.", 1),
    ("Shop có chính sách bảo hành rất tốt.", 1),
    ("Sản phẩm nhỏ gọn, tiện lợi mang theo.", 1),
    ("Chất lượng sản phẩm làm tôi rất ngạc nhiên.", 1),
    ("Hàng giao không đủ số lượng, rất thất vọng.", 0),
    ("Sản phẩm bị méo mó, không sử dụng được.", 0),
    ("Shop không phản hồi khi tôi cần hỗ trợ.", 0),
    ("Sản phẩm không đáng tin cậy.", 0),
    ("Hàng quá hạn sử dụng, không thể dùng.", 0),
    ("Tôi cảm thấy rất thất vọng với dịch vụ.", 0),
    ("Sản phẩm không có tính năng như quảng cáo.", 0),
    ("Giao hàng quá lâu, tôi không hài lòng.", 0),
    ("Tôi đã yêu cầu trả lại sản phẩm này.", 0),
    ("Sản phẩm làm từ chất liệu kém chất lượng.", 0),
    ("Đây là món quà tuyệt vời mà tôi dành cho người thân.", 1),
    ("Hàng giao đúng như cam kết, tôi rất vui.", 1),
    ("Thiết kế sáng tạo, rất độc đáo.", 1),
    ("Chất lượng vượt xa mong đợi, tuyệt vời!", 1),
    ("Shop hỗ trợ bảo hành nhanh chóng, rất hài lòng.", 1),
    ("Sản phẩm này đáng giá đến từng đồng.", 1),
    ("Dịch vụ rất chuyên nghiệp, giao hàng nhanh.", 1),
    ("Tôi đã giới thiệu sản phẩm này cho cả gia đình.", 1),
    ("Mua hàng từ shop này rất đáng tin cậy.", 1),
    ("Hàng chính hãng, chất lượng như quảng cáo.", 1),
    ("Tôi sẽ tiếp tục mua thêm sản phẩm từ shop.", 1),
    ("Sản phẩm quá to, không tiện lợi.", 0),
    ("Thiết kế không vừa ý, tôi rất tiếc.", 0),
    ("Chất lượng sản phẩm không xứng với giá tiền.", 0),
    ("Shop giao nhầm sản phẩm, không chuyên nghiệp.", 0),
    ("Sản phẩm gây nhiều phiền phức khi sử dụng.", 0),
    ("Không thể lắp ráp sản phẩm, rất bực mình.", 0),
    ("Giao hàng sai thời gian, không đúng hẹn.", 0),
    ("Chất lượng sản phẩm làm tôi thất vọng.", 0),
    ("Shop không hỗ trợ trả hàng, rất tệ.", 0),
    ("Tôi sẽ không mua lại từ shop này nữa.", 0),
    ("Sản phẩm có mùi thơm dễ chịu, rất hài lòng.", 1),
    ("Thiết kế độc đáo, chưa từng thấy ở đâu.", 1),
    ("Chất lượng cao cấp, hoàn toàn yên tâm.", 1),
    ("Tôi rất thích cách đóng gói sản phẩm.", 1),
    ("Hàng giao đủ số lượng, đúng như yêu cầu.", 1),
    ("Sản phẩm đẹp hơn mong đợi của tôi.", 1),
    ("Mọi người trong gia đình đều yêu thích sản phẩm này.", 1),
    ("Dịch vụ của shop rất chuyên nghiệp và tận tình.", 1),
    ("Tôi cảm thấy rất hạnh phúc khi sử dụng sản phẩm này.", 1),
    ("Đây là món quà hoàn hảo cho người thân.", 1),
    ("Sản phẩm gây nhiều khó khăn khi sử dụng.", 0),
    ("Tôi không nhận được phụ kiện đi kèm.", 0),
    ("Shop giao nhầm màu sản phẩm, rất thất vọng.", 0),
    ("Hỗ trợ không nhiệt tình, cảm giác rất khó chịu.", 0),
    ("Sản phẩm kém chất lượng, không bền.", 0),
    ("Dịch vụ bảo hành rất chậm, không chuyên nghiệp.", 0),
    ("Tôi đã yêu cầu đổi trả nhưng chưa được xử lý.", 0),
    ("Shop không đáng tin cậy, sẽ không mua lại.", 0),
    ("Hàng không có tem chính hãng, rất nghi ngờ.", 0),
    ("Sản phẩm không đáp ứng được kỳ vọng của tôi.", 0),
    ("Tôi cảm thấy đây là một quyết định mua hàng đúng đắn.", 1),
    ("Chất lượng dịch vụ của shop thật tuyệt vời.", 1),
    ("Hàng giao nhanh, đóng gói rất cẩn thận.", 1),
    ("Sản phẩm rất đẹp, phù hợp với phong cách của tôi.", 1),
    ("Giá cả hợp lý, chất lượng vượt xa mong đợi.", 1),
    ("Tôi sẽ tiếp tục ủng hộ shop trong tương lai.", 1),
    ("Sản phẩm có hướng dẫn sử dụng rất chi tiết.", 1),
    ("Tôi rất hài lòng với sự nhiệt tình của shop.", 1),
    ("Sản phẩm đạt chuẩn chất lượng cao.", 1),
    ("Shop làm việc rất chuyên nghiệp và uy tín.", 1),
]

# Chuyển danh sách thành DataFrame, đổi thứ tự cột
df = pd.DataFrame(reviews, columns=["Text", "Label"])

# Đổi tên cột thành chữ thường
df.columns = df.columns.str.lower()

# Hoán đổi nội dung của hai cột
temp = df['text'].copy()
df['text'] = df['label']
df['label'] = temp

# Sắp xếp lại thứ tự cột để label đứng trước
df = df[["label", "text"]]
# Lưu vào file Excel
df.to_excel("test.xlsx", index=False)

print("Đã lưu nhận xét khách hàng vào file test.xlsx với cột 'text' và 'label'")
